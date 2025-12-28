---
title: "2025: Unwarping, Inferring, Decomposing"
date: Dec 2025
desc: Structure recovery in three registers
---

Three projects this year that turned out to be variations on the same question: how do you recover structure from data that's hiding it?

### 1. Unwarping unwrapped

I "inherited" page-dewarp as Python 2 code and went to some lengths to "inflate" it into proper software. That re-maintenance phase was done mid-2023 but I returned this year and did major renovations.

Quick recap on the cubic sheet model: Matt Zucker's original blog post treats a photographed page as a bent surface rather than a pixel-shuffle problem. You fit cubic Hermite splines to detected text contours, then solve for the coefficients that best explain where the text ended up when projected through your camera model. Physically grounded, geometrically motivated.

Zucker left a tip in his blog post that this was a non-linear least squares optimisation. The original code used SciPy's Powell method (derivative-free), which works but leaves performance on the table. I rewrote the objective in JAX for autodiff, switched to L-BFGS-B, and function evaluations dropped by 100-700x depending on the image. Wall clock: 12 seconds down to under 1.

Batch processing compounds this. JAX's JIT has warmup cost so single images don't see the full benefit, but for a stack of scans (the actual use case) you get 3-5x additional from parallelisation. 40-image batch goes from minutes to seconds.

Minor gripe while I'm here: homography estimation with splines is physically grounded and modern DL research has begun to lay its own Goodhart trap wherein grid-based dewarping methods are privileged. You train a network to predict a pixel displacement field and that's it. These work on their training distribution but they're not modelling actual physics. A page is a surface, it bends according to material properties. The spline approach is grounded, you're recovering geometry not learning a lookup table. I see papers benchmarking on increasingly contrived distortions and I think: nobody is scanning crumpled up pages, we want *books*.

### 2. Detecting the data model

This started as Wikidata preprocessing but became its own thing. Three intertwined packages: genson-core, avrotize-rs, polars-genson.

Having already been a passionate Pydantic modeller, I was attuned to data models. The gap I kept hitting: I have data, I don't have a schema, I need one before I can do anything useful. genson-core infers the schema from JSON (fields, types, optional vs required, mappings). The mappings part was actually the most significant.

I saw this in Wikidata where scalar string fields keyed by language became massive mappings which were mostly null. A field like `labels` that in principle contains a string per language, but in practice most entities have labels in 3-5 languages out of hundreds possible. If you flatten that naively you get hundreds of mostly-null columns. If you recognise it as a map (`Dict[str, str]`) you preserve the structure without the explosion. This is a schema type that can only be represented by map type inference, and I found that genson-rs was unable to do this, so I made my own extension based on it (essentially reprocessing the genson-rs dtype inference).

The CLI tool (genson-cli) exposes it for command-line use. polars-genson wraps it as a Polars plugin so you can do schema inference directly on string columns in a DataFrame.

avrotize-rs handles schema translation. JSON Schema is a good interchange format but Avro is what you actually want for downstream processing (typed, compact, plays well with columnar storage). I ported it from the Python original because I wanted it fast and working with the Rust genson-core without crossing language boundaries.

The three together form a pipeline: raw JSON → inferred JSON Schema → Avro schema → typed processing. What I'm actually doing is constraint discovery. A schema isn't describing what your data *is*, it's describing what your data *isn't allowed to be*.

### 3. Embeddings and their decomposition

#### polars-fastembed

Text embeddings are cool but to me I sensed they'd only be truly useful if they were on my compute platform of choice, Polars. Or as I put it at the time: bring the embeddings to the DataFrame.

The "embeddings as a service" abstraction is unpleasant. They can be computed locally if fast enough, and paying for them is senseless when there are many high quality open source ones. What I wanted was embeddings as a DataFrame operation: column of strings in, column of vectors out, happens locally.

polars-fastembed wraps fastembed-rs. My main contribution to fastembed-rs this year was adding support for Snowflake's Arctic Embed models including the quantised variants. Initial response was "this package doesn't intend to keep feature parity with the Python version" but this just meant "send PRs not issues" so I did.

The performance story matters. My benchmark is embedding all 708 Python PEPs (realistic corpus size, documents vary in length). With the base MiniLM model on CPU this took around a minute. With Arctic Embed XS on GPU via CUDA ONNX runtime, 5 seconds. That's the difference between viable and not.

I also made polars-luxical wrapping Datology's Luxical One model. That one does the same benchmark in under a second on CPU which is some doing. I haven't explored it much but did make polars-luxical to use it.

(There's an existing polars-candle library I wanted to bench against but last I checked it didn't function. Unmaintained I think.)

#### picard-ica

Once you have embeddings you either want to "retrieve" (embedded query → cosine distance → ranked results) or "decompose" for topic modelling. I tried decomposition and found it useful enough that I developed picard-ica.

Embeddings tell you where something is in semantic space (its "meaning"). Decomposition separates out what the dimensions of that space actually are. You can treat it as an optimisation task: how to separate out the components of meaning as you would separate colours of mixed paint or individual vocals in a multi-speaker recording (cocktail party problem).

PICARD stands for Preconditioned ICA for Real Data. The preconditioning refers to whitening (making the covariance spherical, i.e. isotropic baseline variance) and using approximate Hessian to speed up convergence. It handles real world data better than alternatives. Depending on settings the results can be equivalent to FastICA or InfoMax (see the docs at mind-inria/picard).

ICA can be cast as likelihood maximisation for an unmixing matrix W (separating out the sources, in this case the topics). We approximate the 2nd order curvature of the objective function.

Speed is important and there's a cool property here: correctness is required for speed. If you mess any of it up your algo will simply not converge fast. Since the algo ultimately calls LAPACK, I saw this perf as essentially being bottlenecked by correctness. If you get the algo wrong and it hits the 200 max iterations rather than 20 or 30 you can't possibly do that faster no matter how speedy your implementation language might be. In practice it's faster even at the same number of iterations, though I had to walk back some "improvements" which made each iteration slower as overall it wasn't worth the perf hit for a chance at better warmup. My sense is I did it correctly; the convergence step counts don't match because the random number generation engines differ.

I tried a few Rust linear algebra backends. linfa/ndarray performed well. faer is nice in theory (pure Rust) but I saw worse perf and didn't want to take the hit. Looks like development is paused there but the developer is also active in facet-rs so I'll be watching that.

Since picard-ica is a Rust crate I just dropped it into polars-fastembed and it now underlies the topic modelling. Source separation on text embedding vectors directly to give topics. Before I integrated it properly, ICA was a bottleneck in arxiv-explorer. Now it's not.

### The thread

Dewarping recovers geometry from images. Schema inference recovers structure from values. ICA recovers topics from embeddings. Signal hidden in noisy or high-dimensional data, and you extract it.

I keep building the same kind of tool: something that takes data in an inconvenient form and reveals the structure that was already there.
