UV ?= uv
VERSION ?=
TARGET ?= main
PRERELEASE ?= 1
DRY_RUN ?= 0

.PHONY: help check build release-check release dispatch-release sync-local

help:
	@echo "TangoPostureOpenCommunity Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make check             Run ruff, mypy, and headless pytest"
	@echo "  make build             Build wheel and sdist via uv build"
	@echo "  make release-check     Verify clean tree, pyproject version, and quality checks"
	@echo "  make release           Tag, push, and create GitHub Release (triggers CI + downstream PRs)"
	@echo "  make dispatch-release  Manually dispatch release update PRs to websites & Documentation"
	@echo "  make sync-local        Sync sibling ../websites and ../Documentation checkouts locally"
	@echo ""
	@echo "Optional variables: VERSION=0.5.3 TARGET=main PRERELEASE=1 DRY_RUN=1"

check:
	$(UV) run --frozen ruff check src tests
	$(UV) run --frozen mypy src
	QT_QPA_PLATFORM=offscreen $(UV) run --frozen pytest -q

build:
	$(UV) build

release-check:
	$(UV) run python scripts/release.py check \
		$(if $(VERSION),--version "$(VERSION)",) \
		$(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,)

release:
	$(UV) run python scripts/release.py publish \
		--target "$(TARGET)" \
		$(if $(VERSION),--version "$(VERSION)",) \
		$(if $(filter 1 true yes,$(PRERELEASE)),--prerelease,) \
		$(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,)

dispatch-release:
	$(UV) run python .github/scripts/dispatch_release.py \
		--target all \
		--release-version "$(if $(VERSION),$(VERSION),v$(shell $(UV) run python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"))" \
		--prerelease "$(if $(filter 1 true yes,$(PRERELEASE)),true,false)" \
		$(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,)

sync-local:
	$(UV) run python scripts/release.py sync-local \
		$(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,)
