# 0.1.1

Fixed `BatchedBernoulliLogit` losing precision at large logits when compiled with Reactant, and made it return `-Inf` for outcomes other than 0 or 1, as `BernoulliLogit` does.

# 0.1.0

Initial release.
