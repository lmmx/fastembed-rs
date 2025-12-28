# Louis Maddox Writing Style Guide

## Voice and Stance

**Practitioner, not lecturer.** First person throughout. "I found," "I sense," "I went spinning" — claims grounded in direct experience rather than abstracted authority. The credibility comes from having built the thing, not from explaining it.

**Conversational precision.** Technical vocabulary used exactly but not pedantically. Terms like "preconditioning" or "unmixing matrix" appear when they're the right words, but always with enough context that a reader who doesn't know them can follow. No gratuitous jargon, no dumbing down.

**Strategic colloquialism.** Phrases like "begging to be deleted," "this just meant 'send PRs not issues' so I did," "some doing" appear alongside formal technical language. This signals that the author is a person, not a documentation generator.

## Sentence Architecture

**Rhythmic variation.** Short declarative sentences punctuate longer explanatory chains. Example from notes: "Speed is important, cool property here is that correctness is required for speed." Then a longer unpacking follows.

**Parenthetical texture.** Frequent use of parentheses for asides, clarifications, or wry commentary: "(or as I put it at the time, 'bring the embeddings to the dataframe')". These add personality without derailing the main argument.

**Active constructions.** "I made genson-core" not "genson-core was developed." "I saw this in Wikidata" not "this pattern was observed in Wikidata."

## Structural Patterns

**Problems before solutions.** The friction comes first: what was slow, what was broken, what didn't exist. Then the work. Then the implications.

**Numbered or bulleted breakdowns** for complex multi-part explanations. Not as decoration — as cognitive scaffolding.

**Code and concepts interspersed.** Not "here's the explanation, and now here's a code block" — the code appears *within* the flow of argument.

## What to Avoid

**LLM tells:**
- "It's not X — it's Y" inversions
- "Here's the thing:" or "Let me be clear:"
- Unnecessary intensifiers ("incredibly," "absolutely," "truly")
- Faux-profound sentence fragments ("And that changes everything.")
- Symmetric parallel structures that sound like marketing copy
- "In this post, I will..." preambles

**Overclaiming:**
- Assertions of novelty without qualification ("the first," "unprecedented")
- Hyperbolic performance claims without numbers
- Broad claims from narrow evidence
- Treating every project as revolutionary

**False polish:**
- Removing all hedges (keep "I sense," "my read is," "in my experience")
- Eliminating first-draft texture
- Over-editing into blandness

## Distinctive Elements

**Self-correction is visible.** "I used to think X" or "I had to walk back some 'improvements'" — the author's learning process is part of the story.

**Practical humility.** Acknowledges tradeoffs, mentions when something is overkill for simple cases, doesn't pretend every tool is perfect.

**External work cited naturally.** References to papers, other projects, or prior art woven into sentences rather than footnoted or bullet-listed.

**Minimal emoji.** If any, used as section dividers rather than decoration. Never mid-sentence.

## The Arc

A typical piece moves:
1. **Situation**: What existed, what was wrong with it, or what I needed
2. **Work**: What I built or changed, with technical specifics
3. **Observation**: What I learned, what surprised me, what I now think
4. **Implication**: (Optional) What this suggests about the broader problem space

The observation is the payload. The tool is the credential; the insight is the artifact.
