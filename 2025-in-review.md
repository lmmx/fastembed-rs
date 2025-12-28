# 2025: Unwarping, Inferring, Decomposing

This year I worked on three problems that turned out to be the same problem: how do you recover structure from data that's hiding it?

Page images hide their text behind geometric distortion. JSON hides its schema behind its values. Embeddings hide their topics behind 384 dimensions of float soup. The work was: build tools that extract what's actually there.

---

## 1. Unwarping unwrapped

I inherited page-dewarp as Python 2 code back in 2016. Matt Zucker wrote the original as a companion to his blog post on the cubic sheet model — a physically grounded approach to document dewarping that treats the page as a bent surface rather than a pixel-shuffle problem.

By mid-2023 I'd done the maintenance work: Python 3, CLI, pip installable, modular. But the core remained Zucker's original optimisation loop, and that loop was slow. Around 12 seconds per image on my hardware.

This year I went back and did the renovation properly.

The cubic sheet model fits a parametric surface to detected text contours. You're solving for the coefficients of cubic Hermite splines that, when projected through your camera model, best explain where the text ended up. It's non-linear least squares — Zucker noted this in his original post as an "exercise left for the reader."

I took him up on it. The original used SciPy's Powell method, which is derivative-free. Works, but you're leaving performance on the table. I rewrote the objective function in JAX, which gave me autodiff for free, then switched to L-BFGS-B. Function evaluations dropped by 100-700x depending on the image. Wall clock time went from 12+ seconds to under a second.

The batch processing story is its own thing. JAX's JIT compilation means you pay a warmup cost, so single images don't see the full benefit. But when you're processing a stack of scans — which is the actual use case — you get 3-5x additional speedup from parallelisation on top of the per-image gains. For a 40-image batch I was seeing total time drop from minutes to seconds.

One gripe, while I'm here: the deep learning literature on document dewarping has largely moved to grid-based methods. You train a network to predict a pixel displacement field, apply it, done. These work fine on their training distribution, but they're not modelling the actual physics. A page is a surface. It bends according to material properties. The spline-based approach is slower to run (though not anymore, really) but it's *grounded* — you're recovering geometry, not learning a lookup table. I see papers benchmarking on increasingly contrived distortions and I think: nobody is scanning crumpled paper. We want books.

---

## 2. Detecting the data model

The other major project this year was schema inference, which started as Wikidata preprocessing but became its own thing.

Three packages interlock here: **genson-core** for the inference engine, **avrotize-rs** for schema translation, and **polars-genson** for DataFrame integration. The goal: given a pile of JSON, tell me what shape it has.

I'd already spent years as a Pydantic enthusiast, so I was primed to think about data models. But the gap I kept hitting was: I have data, I don't have a schema, and I need one before I can do anything useful. Existing tools would give you field names and basic types. What I needed was richer: optional vs required, nested structure, and critically, *map types*.

The map inference is what made this worth building. I ran into it with Wikidata: you'd have a field like `labels` that in principle contains a string per language. In practice, most entities have labels in 3-5 languages out of hundreds possible. If you flatten that naively you get hundreds of mostly-null columns. If you recognise it as a map — a `Dict[str, str]` — you preserve the structure without the explosion.

genson-rs, which I forked from, couldn't do this. It would see the nested structure but not recognise the map pattern. So I wrote genson-core as an extension layer that reprocesses the type inference and catches these cases. The CLI tool (genson-cli) exposes it for command-line use; polars-genson wraps it as a Polars plugin so you can do schema inference directly on string columns in a DataFrame.

The avrotize-rs piece handles schema translation. JSON Schema is a good interchange format but Avro is what you actually want for downstream processing — it's typed, it's compact, it plays well with columnar storage. avrotize-rs takes a JSON Schema and emits an Avro schema. I ported it from the Python original because I wanted it fast and I wanted it to work with the Rust genson-core without crossing language boundaries.

The three together form a pipeline: raw JSON → inferred JSON Schema → Avro schema → typed processing. What I'm actually building is constraint discovery. A schema isn't describing what your data *is* — it's describing what your data *isn't allowed to be*. Once you have that, you can validate, you can generate types, you can catch drift.

---

## 3. Embeddings and their decomposition

Text embeddings felt like they'd only be useful to me if they lived on my compute platform of choice, Polars. As I put it at the time: bring the embeddings to the DataFrame.

### 3a: polars-fastembed

The "embeddings as a service" abstraction never sat right with me. If the models are open, if inference is fast enough, why am I paying someone to run a forward pass? What I wanted was embeddings as a DataFrame operation — a column of strings goes in, a column of vectors comes out, and it happens locally.

polars-fastembed wraps fastembed-rs, which is itself a clean Rust embedding library. My main contribution to fastembed-rs this year was adding support for Snowflake's Arctic Embed models, including the quantised variants. There was initial pushback ("this package doesn't intend to keep feature parity with the Python version") but that turned out to mean "send PRs not issues," so I did.

The performance story matters here. My benchmark is embedding all 708 Python PEPs — it's a realistic corpus size and the documents vary in length. With the base MiniLM model on CPU, this took around a minute. With Arctic Embed XS on GPU via CUDA ONNX runtime, it dropped to 5 seconds. That's the difference between "I'll run this overnight" and "I'll run this now."

I also made polars-luxical, wrapping Datology's Luxical One model. That one does the same benchmark in under a second *on CPU*. I haven't explored it deeply — the model is newer and less understood — but the speed is striking enough that I wanted it available.

(There's an existing polars-candle library that I wanted to benchmark against, but last I checked it didn't work. Unmaintained, I think.)

### 3b: picard-ica

Once you have embeddings, you can do two things with them: retrieve (query → cosine similarity → ranked results) or decompose (find the latent topics). I wanted both.

ICA — Independent Component Analysis — casts topic modelling as source separation. If embeddings tell you where something is in semantic space, decomposition tells you what the axes of that space actually are. You're separating topics the way you'd separate vocals from a recording with multiple speakers.

I ported PICARD to Rust. PICARD stands for Preconditioned ICA for Real Data — the "preconditioned" part refers to whitening the data (making covariance spherical) and using approximate Hessian information to speed up convergence. It's more robust than FastICA on real-world data, and depending on settings can reproduce FastICA or InfoMax results.

The performance dynamics are interesting. ICA is iterative, and the algorithm has a maximum iteration count (usually 200). If your implementation is correct, you converge in 20-30 iterations. If you've got a bug somewhere — wrong gradient, bad preconditioning — you hit the max. So correctness is required for speed. You can't brute-force your way past a broken algorithm. The LAPACK calls at the bottom are already fast; what matters is not making them run 10x more than necessary.

I tried a few Rust linear algebra backends. linfa/ndarray worked well. faer is appealing in theory (pure Rust, no C dependencies) but I saw worse performance in practice and development there seems paused. (The main faer developer is active in facet-rs, so I'll keep an eye on it.)

The picard-ica crate now underlies topic modelling in polars-fastembed. You embed your documents, call the decomposition method, and get topic assignments. Before I integrated it properly, ICA was a bottleneck in arxiv-explorer. Now it's not.

---

## The common thread

Dewarping recovers geometry from images. Schema inference recovers structure from values. ICA recovers topics from embeddings. These are all the same operation: there's signal hidden in noisy or high-dimensional data, and you want to extract it.

I don't have a grand unified theory. But I notice I keep building the same kind of tool: something that takes data in an inconvenient form and reveals the structure that was always there. Maybe that's just what programming is. Maybe it's a particular disposition. Either way, it's what I'll keep doing.
