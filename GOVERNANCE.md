# Project Governance

## Maintainer

SpikingTSF is maintained by **Jafar Bakhshaliyev** under the [Spikora Neural Research](https://github.com/spikora) organization. The maintainer is responsible for the overall direction of the project, code reviews, and releases.

## Contribution Process

External contributors are welcome through GitHub Issues and Pull Requests.

- **Bug fixes and documentation:** straightforward PRs are reviewed and merged as time permits.
- **New model implementations:** should be discussed in a GitHub Issue before a large pull request is submitted, to confirm alignment with the project scope.
- **Major design changes** (e.g., changes to the training framework, data loading pipeline, or evaluation protocol): must be discussed in a GitHub Issue first. Changes that break reproducibility of existing results require careful justification.

## Decision Making

The maintainer makes final decisions on what is merged. In cases of disagreement, the discussion happens openly in the relevant issue or pull request.

## Contributor Recognition

Contributors who make significant additions (new models, verified results, infrastructure improvements) may be acknowledged in `ACKNOWLEDGEMENTS.md` and in release notes.

## Versioning

The project follows semantic versioning (MAJOR.MINOR.PATCH). Minor releases add new models or datasets. Patch releases fix bugs or improve documentation. The roadmap is in [ROADMAP.md](ROADMAP.md).
