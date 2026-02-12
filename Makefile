.PHONY: sync cmd alias config

SSH_HOST ?= NNU
REMOTE_DIR ?= /home/user/code/mRAG
LOCAL_DIR ?= $(CURDIR)
EXCLUDE_FILE ?= .exclude
ALIAS_FILE ?= .alias
SSH_KEY ?= $(HOME)/.ssh/id_rsa.pub
CONDA_ENV ?= /home/user/env/envs/py310
CONDA_BIN ?= conda
CMD ?=
KNOWN_TARGETS := sync cmd alias config
CMD_GOALS := $(filter-out $(KNOWN_TARGETS),$(MAKECMDGOALS))

ifneq ($(filter cmd,$(MAKECMDGOALS)),)
ifneq ($(strip $(CMD_GOALS)),)
$(eval $(CMD_GOALS):;@:)
endif
endif



sync:
	rsync -azP --delete \
		--exclude-from='$(EXCLUDE_FILE)' \
		-e ssh \
		'$(LOCAL_DIR)/' \
		'$(SSH_HOST):$(REMOTE_DIR)/'

# 在远程执行命令:
# 1) make cmd CMD='nvidia-smi'
# 2) make cmd   # 无 CMD 时自动取本地历史上一条命令
cmd:
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
