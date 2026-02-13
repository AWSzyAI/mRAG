.PHONY: sync cmd alias config bpe clean

# =====================================================================
# ┌─────────────────┐          rsync           ┌──────────────────┐
# │    本地 Mac   	 │ ──────────────────────→	│      服务器       │
# │  - codex	   	│                      	    │   - GPU 运行     │
# │  - Claude Code  │ ←──────────────────────  	│   - 模型推理      │
# └─────────────────┘        结果拉取          	 └──────────────────┘
# =====================================================================

# SYNC_HOST ?= Ocean-NAT
# SYNC_HOST ?= featurize
SYNC_HOST ?= AC


# =====================================================================
SYNC_FILE ?= .sync_ssh
SSH_HOST ?= $(SYNC_HOST)
REMOTE_DIR ?= $(strip $(shell awk -v host='$(SYNC_HOST)' -v key='REMOTE_DIR' -f scripts/read_sync_value.awk '$(SYNC_FILE)' 2>/dev/null))
CONDA_ENV ?= $(strip $(shell awk -v host='$(SYNC_HOST)' -v key='CONDA_ENV' -f scripts/read_sync_value.awk '$(SYNC_FILE)' 2>/dev/null))
# =====================================================================

LOCAL_DIR ?= $(CURDIR)
EXCLUDE_FILE ?= .exclude
ALIAS_FILE ?= .alias
SSH_KEY ?= $(HOME)/.ssh/id_rsa.pub
BPE_FILE ?= models/bpe_simple_vocab_16e6.txt.gz
BPE_NAME ?= bpe_simple_vocab_16e6.txt.gz

CONDA_BIN ?= conda
CMD ?=
KNOWN_TARGETS := sync cmd alias config bpe clean
CMD_GOALS := $(filter-out $(KNOWN_TARGETS),$(MAKECMDGOALS))

ifneq ($(filter cmd,$(MAKECMDGOALS)),)
ifneq ($(strip $(CMD_GOALS)),)
$(eval $(CMD_GOALS):;@:)
endif
endif


define ensure_remote_sync_config
	@if [ -z "$(SSH_HOST)" ] || [ -z "$(REMOTE_DIR)" ] || [ -z "$(CONDA_ENV)" ]; then \
		echo "Missing remote sync config."; \
		echo "SYNC_HOST=$(SYNC_HOST) SSH_HOST=$(SSH_HOST) REMOTE_DIR=$(REMOTE_DIR) CONDA_ENV=$(CONDA_ENV)"; \
		echo "Please check $(SYNC_FILE), expected block:"; \
		echo "  Host $(SYNC_HOST)"; \
		echo "    REMOTE_DIR ?= ..."; \
		echo "    CONDA_ENV ?= ..."; \
		exit 2; \
	fi
endef



sync:
	$(ensure_remote_sync_config)
	@set -e; \
	filter_file="$$(mktemp)"; \
	trap 'rm -f "$$filter_file"' EXIT; \
	awk -f scripts/build_rsync_filter.awk '$(EXCLUDE_FILE)' > "$$filter_file"; \
	rsync -azP --delete \
		--filter="merge $$filter_file" \
		-e ssh \
		'$(LOCAL_DIR)/' \
		'$(SSH_HOST):$(REMOTE_DIR)/'

# 在远程执行命令:
# 1) make cmd CMD='nvidia-smi'
# 2) make cmd   # 无 CMD 时自动取本地历史上一条命令
cmd:
	$(ensure_remote_sync_config)
	@set -e; \
	LC_ALL=C; export LC_ALL; \
	run_cmd='$(CMD)'; \
	if [ -z "$$run_cmd" ]; then \
		run_cmd='$(CMD_GOALS)'; \
	fi; \
	if [ -z "$$run_cmd" ]; then \
		hist_file="$${HISTFILE:-$$HOME/.zsh_history}"; \
		if [ -f "$$hist_file" ]; then \
			run_cmd="$$(tail -n 500 "$$hist_file" \
				| tr -d '\000' \
				| sed -E 's/^: [0-9]+:[0-9]+;//' \
				| sed -E 's/^[[:space:]]+//; s/[[:space:]]+$$//' \
				| sed -E '/^$$/d' \
				| sed -E '/^make([[:space:]].*)?cmd([[:space:]]|$$)/d' \
				| sed -E '/^mc([[:space:]]|$$)/d' \
				| tail -n 1)"; \
		fi; \
	fi; \
	if [ -z "$$run_cmd" ]; then \
		echo "Usage: make cmd CMD='<remote command>'"; \
		echo "Hint: failed to detect previous command from history."; \
		exit 2; \
	fi; \
	escaped_cmd="$$(printf "%s" "$$run_cmd" | sed "s/'/'\"'\"'/g")"; \
	echo "Remote CMD: $$run_cmd"; \
	ssh $(SSH_HOST) "bash -lc 'cd $(REMOTE_DIR) && eval \"\$$($(CONDA_BIN) shell.bash hook)\" && conda activate $(CONDA_ENV) && $$escaped_cmd'"

# 根据 .alias 生成 shell alias，建议: eval "$$(make -s alias)"
alias:
	@printf '%s\n' \
		'unalias mc 2>/dev/null || true' \
		'function mc {' \
		'  local __mrag_prev=""' \
		'  if [ "$$#" -gt 0 ]; then' \
		'    make cmd CMD="$$*"' \
		'    return $$?' \
		'  fi' \
		'  if [ -n "$${ZSH_VERSION-}" ]; then' \
		'    __mrag_prev="$${history[$$HISTCMD]}"' \
		'    case "$$__mrag_prev" in' \
		'      ""|mc|mc\ *|make\ cmd|make\ cmd\ *) __mrag_prev="$${history[$$((HISTCMD-1))]}" ;;' \
		'    esac' \
		'  fi' \
		'  if [ -z "$$__mrag_prev" ]; then' \
		'    __mrag_prev="$$(fc -ln -1 2>/dev/null | sed -E '\''s/^[[:space:]]+//; s/[[:space:]]+$$//'\'' | tail -n 1)"' \
		'  fi' \
		'  case "$$__mrag_prev" in' \
		'    ""|mc|mc\ *|make\ cmd|make\ cmd\ *)' \
		'      __mrag_prev="$$(fc -ln -2 2>/dev/null | sed -E '\''s/^[[:space:]]+//; s/[[:space:]]+$$//'\'' | tail -n 1)"' \
		'      ;;' \
		'  esac' \
		'  if [ -z "$$__mrag_prev" ]; then' \
		'    echo "mc: failed to resolve previous command"' \
		'    return 2' \
		'  fi' \
		'  make cmd CMD="$$__mrag_prev"' \
		'}'
	@awk -F':' '\
		function trim(s){ sub(/^[[:space:]]+/, "", s); sub(/[[:space:]]+$$/, "", s); return s } \
		/^[[:space:]]*$$/ { next } \
		/^[[:space:]]*#/ { next } \
		/^[[:space:]]*\/\// { next } \
		{ \
			name = trim($$1); \
			cmd = trim(substr($$0, index($$0, ":") + 1)); \
			if (name == "" || cmd == "") next; \
			if (name == "mc") next; \
			gsub(/\047/, "'\''\"'\''\"'\''", cmd); \
			printf("alias %s='\''%s'\''\n", name, cmd); \
		} \
	' $(ALIAS_FILE)

config:
	$(ensure_remote_sync_config)
	@if [ ! -f "$(SSH_KEY)" ]; then \
		echo "SSH key not found: $(SSH_KEY)"; \
		exit 2; \
	fi
	ssh-copy-id -i $(SSH_KEY) $(SSH_HOST)
	@set -e; \
	rc_file="$(HOME)/.zshrc"; \
	begin="# >>> mRAG alias init >>>"; \
	end="# <<< mRAG alias init <<<"; \
	tmp_file="$$(mktemp)"; \
	if [ -f "$$rc_file" ]; then \
		awk -v b="$$begin" -v e="$$end" '\
			$$0 == b { skip = 1; next } \
			$$0 == e { skip = 0; next } \
			skip != 1 { print } \
		' "$$rc_file" > "$$tmp_file"; \
	else \
		: > "$$tmp_file"; \
	fi; \
	{ \
		cat "$$tmp_file"; \
		printf '\n%s\n' "$$begin"; \
		printf 'if [ -f "%s/Makefile" ]; then\n' '$(LOCAL_DIR)'; \
		printf '  eval "$$(make -s -C %s alias 2>/dev/null)"\n' '$(LOCAL_DIR)'; \
		printf 'fi\n'; \
		printf '%s\n' "$$end"; \
	} > "$$rc_file"; \
		rm -f "$$tmp_file"; \
		echo "Updated $$rc_file with mRAG alias bootstrap."; \
		echo "To use mc/ms in this current shell now, run: eval \"\$$(make -s alias)\""

bpe:
	$(ensure_remote_sync_config)
	@set -e; \
	mkdir -p models; \
	local_target="$(BPE_FILE)"; \
	local_src1="github/LLaVA-NeXT/llava/model/multimodal_encoder/dev_eva_clip/eva_clip/$(BPE_NAME)"; \
	local_src2="$$HOME/.cache/scenic/clip/$(BPE_NAME)"; \
	if [ -f "$$local_target" ]; then \
		echo "Local BPE exists: $$local_target"; \
	elif [ -f "$$local_src1" ]; then \
		cp "$$local_src1" "$$local_target"; \
		echo "Local BPE copied: $$local_src1 -> $$local_target"; \
	elif [ -f "$$local_src2" ]; then \
		cp "$$local_src2" "$$local_target"; \
		echo "Local BPE copied: $$local_src2 -> $$local_target"; \
	else \
		echo "Local BPE source not found. Skipped local copy."; \
	fi; \
	ssh $(SSH_HOST) "bash -lc 'set -e; \
		remote_target=\"$(REMOTE_DIR)/models/$(BPE_NAME)\"; \
		remote_src1=\"$(REMOTE_DIR)/github/LLaVA-NeXT/llava/model/multimodal_encoder/dev_eva_clip/eva_clip/$(BPE_NAME)\"; \
		remote_src2=\"\$$HOME/.cache/scenic/clip/$(BPE_NAME)\"; \
		mkdir -p \"$(REMOTE_DIR)/models\"; \
		if [ -f \"\$$remote_target\" ]; then \
			echo \"Remote BPE exists: \$$remote_target\"; \
		elif [ -f \"\$$remote_src1\" ]; then \
			cp \"\$$remote_src1\" \"\$$remote_target\"; \
			echo \"Remote BPE copied: \$$remote_src1 -> \$$remote_target\"; \
		elif [ -f \"\$$remote_src2\" ]; then \
			cp \"\$$remote_src2\" \"\$$remote_target\"; \
			echo \"Remote BPE copied: \$$remote_src2 -> \$$remote_target\"; \
		else \
			echo \"Remote BPE source not found. Run tokenizer once online or place file manually.\"; \
			exit 2; \
		fi'"

clean:
	@set -e; \
	echo "Cleaning Python cache files..."; \
	find . \
		\( -path "./.git" -o -path "./.venv" \) -prune -o \
		-type d \
		\( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name ".ipynb_checkpoints" \) \
		-prune -exec rm -rf {} +; \
	find . \
		\( -path "./.git" -o -path "./.venv" \) -prune -o \
		-type f \
		\( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) \
		-delete; \
	echo "Clean complete."
