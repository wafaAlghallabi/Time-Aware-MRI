# Step generation

Provider-specific runners generate reasoning steps and final answers for Table 1. Each image entry must preserve `timepoint`, `sequence`, `view`, `path`, and `filename` metadata. The generation scripts use `sequence` and `view` when constructing multi-view inputs.
