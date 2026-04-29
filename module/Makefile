.DEFAULT_GOAL := help

.PHONY: help sync sync-code pull pull-role cmd alias config role-local role-remote role-show bpe clean

PROJECT_ROOT ?= $(CURDIR)
MODULE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
MODULE_NAME ?= $(notdir $(MODULE_DIR))

# =====================================================================
# ┌─────────────────┐          rsync           ┌──────────────────┐
# │    本地 Mac     │ ───────────────────────→ │      服务器      │
# │  - codex        │ ←─────────────────────── │   - GPU 运行     │
# │  - Claude Code  │        结果拉取          │   - 模型推理     │
# └─────────────────┘                          └──────────────────┘
# =====================================================================

# SYNC_HOST ?= nnu
# SYNC_HOST ?= AB
# SYNC_HOST ?= featurize
# SYNC_HOST ?= AC
# SYNC_HOST ?= zhichuanyun

SYNC_FILE ?= $(MODULE_DIR)/.sync_ssh
EXCLUDE_FILE ?= $(MODULE_DIR)/.exclude
ALIAS_FILE ?= $(MODULE_DIR)/.alias
PULL_DEFAULT_LIST ?= $(MODULE_DIR)/pull_list.txt
RESULT_PULL_LIST ?= $(MODULE_DIR)/result.txt
READ_SYNC_AWK ?= $(MODULE_DIR)/scripts/read_sync_value.awk
BUILD_RSYNC_FILTER_AWK ?= $(MODULE_DIR)/scripts/build_rsync_filter.awk
ROLE_PROMPT_FILE ?= $(MODULE_DIR)/templates/role_prompt.md

SSH_HOST ?= $(SYNC_HOST)
REMOTE_DIR ?= $(strip $(shell awk -v host='$(SYNC_HOST)' -v key='REMOTE_DIR' -f '$(READ_SYNC_AWK)' '$(SYNC_FILE)' 2>/dev/null))
CONDA_ENV ?= $(strip $(shell awk -v host='$(SYNC_HOST)' -v key='CONDA_ENV' -f '$(READ_SYNC_AWK)' '$(SYNC_FILE)' 2>/dev/null))

LOCAL_DIR ?= $(PROJECT_ROOT)
SSH_KEY ?= $(HOME)/.ssh/id_rsa.pub
BPE_FILE ?= $(PROJECT_ROOT)/models/bpe_simple_vocab_16e6.txt.gz
BPE_NAME ?= bpe_simple_vocab_16e6.txt.gz

CONDA_BIN ?= conda
CMD ?=
KNOWN_TARGETS := help sync sync-code pull pull-role cmd alias config role-local role-remote role-show bpe clean
CMD_GOALS := $(filter-out $(KNOWN_TARGETS),$(MAKECMDGOALS))

ifneq ($(filter cmd,$(MAKECMDGOALS)),)
ifneq ($(strip $(CMD_GOALS)),)
$(eval $(CMD_GOALS):;@:)
endif
endif

ifneq ($(filter pull,$(MAKECMDGOALS)),)
ifneq ($(strip $(CMD_GOALS)),)
$(eval $(CMD_GOALS):;@:)
endif
endif

ifneq ($(filter sync,$(MAKECMDGOALS)),)
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

define resolve_named_pull_list
	@if [ -f "$$1" ]; then \
		printf '%s\n' "$$1"; \
	elif [ -f "$$1.txt" ]; then \
		printf '%s\n' "$$1.txt"; \
	elif [ -f "$(MODULE_DIR)/$$1" ]; then \
		printf '%s\n' "$(MODULE_DIR)/$$1"; \
	elif [ -f "$(MODULE_DIR)/$$1.txt" ]; then \
		printf '%s\n' "$(MODULE_DIR)/$$1.txt"; \
	else \
		printf '%s\n' "$$1"; \
	fi
endef

help:
	@echo "Available targets:"; \
	echo "  make help                Show this menu"; \
	echo "  make sync                Preview sync local -> remote"; \
	echo "  make sync y              Apply sync local -> remote"; \
	echo "  make pull                Preview pull using $(PULL_DEFAULT_LIST)"; \
	echo "  make pull <name>         Preview pull using <name> or <name>.txt"; \
	echo "  make pull y              Apply pull using $(PULL_DEFAULT_LIST)"; \
	echo "  make cmd CMD='<remote command>'"; \
	echo "  make alias               Show alias config"; \
	echo "  make config              Install alias bootstrap + role metadata"; \
	echo "  make role-show           Show local/remote role"; \
	echo "  make bpe                 Upload BPE file"; \
	echo "  make clean               Clean generated files"; \
	echo ""; \
	echo "Sync module files:"; \
	echo "  $(MODULE_DIR)/.sync_ssh"; \
	echo "  $(MODULE_DIR)/.exclude"; \
	echo "  $(MODULE_DIR)/.alias"; \
	echo "  $(MODULE_DIR)/pull_list.txt"; \
	echo "  $(MODULE_DIR)/result.txt"

sync:
	$(ensure_remote_sync_config)
	@set -e; \
	sync_args='$(CMD_GOALS)'; \
	apply_mode=0; \
	if [ "$(APPLY)" = "1" ] || [ "$(Y)" = "1" ]; then apply_mode=1; fi; \
	for arg in $$sync_args; do \
		case "$$arg" in \
			y|yes|--yes|-y) apply_mode=1 ;; \
		esac; \
	done; \
	if [ "$$apply_mode" -eq 1 ]; then \
		echo "Mode: APPLY (sync local -> remote)"; \
		$(MAKE) sync-code APPLY=1; \
	else \
		echo "Mode: PREVIEW (dry-run only)"; \
		$(MAKE) sync-code; \
	fi

sync-code:
	$(ensure_remote_sync_config)
	@set -e; set -o pipefail; \
	filter_file="$$(mktemp)"; \
	tmp_log="$$(mktemp)"; \
	tmp_changes="$$(mktemp)"; \
	tmp_upload="$$(mktemp)"; \
	trap 'rm -f "$$filter_file" "$$tmp_log" "$$tmp_changes" "$$tmp_upload"' EXIT; \
	awk -f "$(BUILD_RSYNC_FILTER_AWK)" "$(EXCLUDE_FILE)" > "$$filter_file"; \
	rsync_args='-az --partial --delete --itemize-changes --stats --human-readable -e ssh'; \
	if [ "$(APPLY)" = "1" ]; then \
		if ! rsync $$rsync_args --filter="merge $$filter_file" \
			'$(LOCAL_DIR)/' \
			'$(SSH_HOST):$(REMOTE_DIR)/' > "$$tmp_log" 2>&1; then \
			echo "[ERROR] rsync failed. Last lines:"; \
			tail -n 40 "$$tmp_log"; \
			exit 1; \
		fi; \
	else \
		if ! rsync $$rsync_args --filter="merge $$filter_file" --dry-run \
			'$(LOCAL_DIR)/' \
			'$(SSH_HOST):$(REMOTE_DIR)/' > "$$tmp_log" 2>&1; then \
			echo "[ERROR] rsync failed. Last lines:"; \
			tail -n 40 "$$tmp_log"; \
			exit 1; \
		fi; \
	fi; \
	echo "---- changes (git style) ----"; \
	LC_ALL=C awk '\
		function is_itemized(tok) { \
			return (tok ~ /^[<>ch\.][fdLDS][^ ]{10}$$/); \
		} \
		/^\*deleting[[:space:]]+/ { \
			path = $$0; sub(/^\*deleting[[:space:]]+/, "", path); \
			printf("-\t%s\n", path); \
			next; \
		} \
		{ \
			item = $$1; \
			if (!is_itemized(item)) next; \
			path = $$0; sub(/^[^ ]+[[:space:]]+/, "", path); \
			if (substr(item, 1, 1) == ".") next; \
			op = "~"; \
			if (item ~ /\+\+\+\+\+\+\+\+\+/) op = "+"; \
			printf("%s\t%s\n", op, path); \
		} \
	' "$$tmp_log" > "$$tmp_changes"; \
	color_pref="$${MS_COLOR:-always}"; \
	color_enabled=0; \
	case "$$color_pref" in \
		always|1|true|TRUE) color_enabled=1 ;; \
		never|0|false|FALSE) color_enabled=0 ;; \
		*) \
			if [ -n "$${NO_COLOR-}" ] || [ -z "$${TERM-}" ] || [ "$${TERM}" = "dumb" ] || [ ! -t 1 ]; then \
				color_enabled=0; \
			else \
				color_enabled=1; \
			fi ;; \
	esac; \
	c_red=''; c_green=''; c_yellow=''; c_reset=''; \
	if [ "$$color_enabled" -eq 1 ]; then \
		c_red="$$(printf '\033[31m')"; \
		c_green="$$(printf '\033[32m')"; \
		c_yellow="$$(printf '\033[33m')"; \
		c_reset="$$(printf '\033[0m')"; \
	fi; \
	while IFS=$$(printf '\t') read -r op rel; do \
		[ -z "$$op" ] && continue; \
		abs='$(LOCAL_DIR)'/"$$rel"; \
		if [ -f "$$abs" ]; then \
			b="$$(wc -c < "$$abs" | tr -d '[:space:]')"; \
			size="$$(awk -v b="$$b" 'BEGIN { \
				if (b >= 1024*1024*1024) printf "%.2fG", b/(1024*1024*1024); \
				else if (b >= 1024*1024) printf "%.2fM", b/(1024*1024); \
				else if (b >= 1024) printf "%.2fK", b/1024; \
				else printf "%dB", b; \
			}')"; \
		else \
			size="N/A"; \
		fi; \
		if [ "$$color_enabled" -eq 1 ]; then \
			c="$$c_yellow"; \
			case "$$op" in \
				+) c="$$c_green" ;; \
				-) c="$$c_red" ;; \
			esac; \
			printf "%b%s%b %b%s%b (%s)\n" "$$c" "$$op" "$$c_reset" "$$c" "$$rel" "$$c_reset" "$$size"; \
		else \
			echo "$$op $$rel ($$size)"; \
		fi; \
	done < "$$tmp_changes"; \
	change_count="$$(wc -l < "$$tmp_changes" | tr -d '[:space:]')"; \
	LC_ALL=C awk '\
		function is_upload_file(tok) { \
			return (tok ~ /^[<>ch]f[^ ]{10}$$/); \
		} \
		{ \
			item = $$1; \
			if (!is_upload_file(item)) next; \
			path = $$0; sub(/^[^ ]+[[:space:]]+/, "", path); \
			print path; \
		} \
	' "$$tmp_log" | sort -u > "$$tmp_upload"; \
	files_changed=0; \
	total_bytes=0; \
	total_lines=0; \
	while IFS= read -r rel; do \
		[ -z "$$rel" ] && continue; \
		abs='$(LOCAL_DIR)'/"$$rel"; \
		if [ -f "$$abs" ]; then \
			b="$$(wc -c < "$$abs" | tr -d '[:space:]')"; \
			l="$$(wc -l < "$$abs" | tr -d '[:space:]')"; \
			total_bytes=$$((total_bytes + b)); \
			total_lines=$$((total_lines + l)); \
			files_changed=$$((files_changed + 1)); \
		fi; \
	done < "$$tmp_upload"; \
	human_size="$$(awk -v b="$$total_bytes" 'BEGIN { \
		if (b >= 1024*1024*1024) printf "%.2fG", b/(1024*1024*1024); \
		else if (b >= 1024*1024) printf "%.2fM", b/(1024*1024); \
		else if (b >= 1024) printf "%.2fK", b/1024; \
		else printf "%dB", b; \
	}')"; \
	rsync_file_count="$$(LC_ALL=C awk -F': ' '/^Number of regular files transferred:/ {print $$2; exit}' "$$tmp_log")"; \
	rsync_tx_size="$$(LC_ALL=C awk -F': ' '/^Total transferred file size:/ {print $$2; exit}' "$$tmp_log")"; \
	delete_warn_count="$$(grep -c '^cannot delete non-empty directory:' "$$tmp_log" || true)"; \
	[ -z "$$rsync_file_count" ] && rsync_file_count=0; \
	[ -z "$$rsync_tx_size" ] && rsync_tx_size=0B; \
	echo "---- summary ----"; \
	echo "Changed entries: $$change_count"; \
	echo "Files to upload: $$files_changed"; \
	echo "Code volume (upload files): $$human_size ($$total_bytes bytes)"; \
	echo "Code lines (upload files): $$total_lines"; \
	echo "Rsync regular files transferred: $$rsync_file_count"; \
	echo "Rsync transferred file size: $$rsync_tx_size"; \
	if [ "$$delete_warn_count" -gt 0 ]; then \
		echo "[WARN] Failed directory deletions: $$delete_warn_count (directory still non-empty on remote)"; \
		grep '^cannot delete non-empty directory:' "$$tmp_log" | sed -n '1,5p' | sed 's/^/[WARN] /'; \
	fi

pull:
	$(ensure_remote_sync_config)
	@set -e; \
	pull_args='$(CMD_GOALS)'; \
	list_file='$(PULL_DEFAULT_LIST)'; \
	apply_mode=0; \
	if [ "$(APPLY)" = "1" ] || [ "$(Y)" = "1" ]; then apply_mode=1; fi; \
	for arg in $$pull_args; do \
		case "$$arg" in \
			y|yes|--yes|-y) apply_mode=1 ;; \
			result) list_file='$(RESULT_PULL_LIST)' ;; \
			*.txt) list_file="$$( \
				if [ -f "$$arg" ]; then printf '%s\n' "$$arg"; \
				elif [ -f "$(MODULE_DIR)/$$arg" ]; then printf '%s\n' "$(MODULE_DIR)/$$arg"; \
				else printf '%s\n' "$$arg"; fi \
			)" ;; \
			*) list_file="$$( \
				if [ -f "$$arg.txt" ]; then printf '%s\n' "$$arg.txt"; \
				elif [ -f "$(MODULE_DIR)/$$arg.txt" ]; then printf '%s\n' "$(MODULE_DIR)/$$arg.txt"; \
				elif [ -f "$$arg" ]; then printf '%s\n' "$$arg"; \
				elif [ -f "$(MODULE_DIR)/$$arg" ]; then printf '%s\n' "$(MODULE_DIR)/$$arg"; \
				else printf '%s\n' "$$arg"; fi \
			)" ;; \
		esac; \
	done; \
	if [ ! -f "$$list_file" ]; then \
		echo "Pull list not found: $$list_file"; \
		echo "Usage:"; \
		echo "  make pull                # preview using $(PULL_DEFAULT_LIST)"; \
		echo "  make pull result         # preview using $(RESULT_PULL_LIST)"; \
		echo "  make pull y              # apply using $(PULL_DEFAULT_LIST)"; \
		echo "  make pull result y       # apply using $(RESULT_PULL_LIST)"; \
		echo "  make pull APPLY=1        # apply using $(PULL_DEFAULT_LIST)"; \
		echo "Note: use trailing 'y' (not '-y')."; \
		exit 2; \
	fi; \
	tmp_list="$$(mktemp)"; \
	tmp_list_main="$$(mktemp)"; \
	tmp_list_log="$$(mktemp)"; \
	tmp_log_main="$$(mktemp)"; \
	tmp_log_log="$$(mktemp)"; \
	tmp_log_all="$$(mktemp)"; \
	tmp_changes="$$(mktemp)"; \
	tmp_pull="$$(mktemp)"; \
	tmp_pull_log="$$(mktemp)"; \
	tmp_log_conflict="$$(mktemp)"; \
	trap 'rm -f "$$tmp_list" "$$tmp_list_main" "$$tmp_list_log" "$$tmp_log_main" "$$tmp_log_log" "$$tmp_log_all" "$$tmp_changes" "$$tmp_pull" "$$tmp_pull_log" "$$tmp_log_conflict"' EXIT; \
	awk '\
		{ line=$$0; sub(/\r$$/, "", line); } \
		/^[[:space:]]*$$/ { next } \
		/^[[:space:]]*#/ { next } \
		{ gsub(/^[[:space:]]+|[[:space:]]+$$/, "", line); print line } \
	' "$$list_file" > "$$tmp_list"; \
	if [ ! -s "$$tmp_list" ]; then \
		echo "Pull list is empty after filtering comments/blanks: $$list_file"; \
		exit 2; \
	fi; \
	: > "$$tmp_list_main"; \
	: > "$$tmp_list_log"; \
	while IFS= read -r rel; do \
		[ -z "$$rel" ] && continue; \
		rel="$${rel#./}"; \
		case "$$rel" in \
			log/*) printf '%s\n' "$$rel" >> "$$tmp_list_log" ;; \
			*) printf '%s\n' "$$rel" >> "$$tmp_list_main" ;; \
		esac; \
	done < "$$tmp_list"; \
	echo "Pull list: $$list_file"; \
	if [ "$$apply_mode" -eq 1 ]; then \
		echo "Mode: APPLY (download changes)"; \
	else \
		echo "Mode: PREVIEW (dry-run only)"; \
	fi; \
	rsync_args='-azPr --itemize-changes --stats --ignore-missing-args -e ssh --files-from'; \
	if [ -s "$$tmp_list_main" ]; then \
		if [ "$$apply_mode" -eq 0 ]; then \
			if ! rsync $$rsync_args "$$tmp_list_main" --dry-run "$(SSH_HOST):$(REMOTE_DIR)/" "$(LOCAL_DIR)/" > "$$tmp_log_main" 2>&1; then \
				echo "[ERROR] rsync failed. Last lines:"; \
				tail -n 40 "$$tmp_log_main"; \
				exit 1; \
			fi; \
		else \
			if ! rsync $$rsync_args "$$tmp_list_main" "$(SSH_HOST):$(REMOTE_DIR)/" "$(LOCAL_DIR)/" > "$$tmp_log_main" 2>&1; then \
				echo "[ERROR] rsync failed. Last lines:"; \
				tail -n 40 "$$tmp_log_main"; \
				exit 1; \
			fi; \
		fi; \
	else \
		: > "$$tmp_log_main"; \
	fi; \
	if [ -s "$$tmp_list_log" ]; then \
		if ! rsync $$rsync_args "$$tmp_list_log" --dry-run "$(SSH_HOST):$(REMOTE_DIR)/" "$(LOCAL_DIR)/" > "$$tmp_log_log" 2>&1; then \
			echo "[ERROR] rsync failed. Last lines:"; \
			tail -n 40 "$$tmp_log_log"; \
			exit 1; \
		fi; \
	else \
		: > "$$tmp_log_log"; \
	fi; \
	cat "$$tmp_log_main" "$$tmp_log_log" > "$$tmp_log_all"; \
	awk '\
		function is_pull_file(tok) { \
			return (tok ~ /^[<>ch]f[^ ]{10}$$/); \
		} \
		{ \
			item = $$1; \
			if (!is_pull_file(item)) next; \
			path = $$0; sub(/^[^ ]+[[:space:]]+/, "", path); \
			print path; \
		} \
	' "$$tmp_log_log" | sort -u > "$$tmp_pull_log"; \
	log_plan_count="$$(wc -l < "$$tmp_pull_log" | tr -d '[:space:]')"; \
	log_conflict_plan=0; \
	: > "$$tmp_log_conflict"; \
	while IFS= read -r rel; do \
		[ -z "$$rel" ] && continue; \
		abs='$(LOCAL_DIR)'/"$$rel"; \
		if [ -e "$$abs" ]; then \
			log_conflict_plan=$$((log_conflict_plan + 1)); \
			printf '%s\n' "$$rel" >> "$$tmp_log_conflict"; \
		fi; \
	done < "$$tmp_pull_log"; \
	log_new_plan=$$((log_plan_count - log_conflict_plan)); \
	log_new_added=0; \
	log_conflict_added=0; \
	if [ "$$apply_mode" -eq 1 ] && [ -s "$$tmp_pull_log" ]; then \
		stamp="$$(date +%Y%m%d_%H%M%S)"; \
		while IFS= read -r rel; do \
			[ -z "$$rel" ] && continue; \
			src="$(SSH_HOST):$(REMOTE_DIR)/$$rel"; \
			dst='$(LOCAL_DIR)'/"$$rel"; \
			mkdir -p "$$(dirname "$$dst")"; \
			if grep -Fxq "$$rel" "$$tmp_log_conflict"; then \
				conflict_dst="$$dst.from_$(SYNC_HOST)_$$stamp"; \
				n=1; \
				while [ -e "$$conflict_dst" ]; do \
					conflict_dst="$$dst.from_$(SYNC_HOST)_$$stamp_$$n"; \
					n=$$((n + 1)); \
				done; \
				if ! rsync -azP -e ssh "$$src" "$$conflict_dst" > /dev/null 2>&1; then \
					echo "[ERROR] rsync failed for log conflict copy: $$rel"; \
					exit 1; \
				fi; \
				log_conflict_added=$$((log_conflict_added + 1)); \
			else \
				if ! rsync -azP -e ssh "$$src" "$$dst" > /dev/null 2>&1; then \
					echo "[ERROR] rsync failed for log file: $$rel"; \
					exit 1; \
				fi; \
				log_new_added=$$((log_new_added + 1)); \
			fi; \
		done < "$$tmp_pull_log"; \
	fi; \
	echo "---- changes (git style) ----"; \
	awk '\
		function is_itemized(tok) { \
			return (tok ~ /^[<>ch\.][fdLDS][^ ]{10}$$/); \
		} \
		/^\*deleting[[:space:]]+/ { \
			path = $$0; sub(/^\*deleting[[:space:]]+/, "", path); \
			printf("-\t%s\n", path); \
			next; \
		} \
		{ \
			item = $$1; \
			if (!is_itemized(item)) next; \
			path = $$0; sub(/^[^ ]+[[:space:]]+/, "", path); \
			if (substr(item, 1, 1) == ".") next; \
			op = "~"; \
			if (item ~ /\+\+\+\+\+\+\+\+\+/) op = "+"; \
			printf("%s\t%s\n", op, path); \
		} \
	' "$$tmp_log_all" > "$$tmp_changes"; \
	color_pref="$${MS_COLOR:-always}"; \
	color_enabled=0; \
	case "$$color_pref" in \
		always|1|true|TRUE) color_enabled=1 ;; \
		never|0|false|FALSE) color_enabled=0 ;; \
		*) \
			if [ -n "$${NO_COLOR-}" ] || [ -z "$${TERM-}" ] || [ "$${TERM}" = "dumb" ] || [ ! -t 1 ]; then \
				color_enabled=0; \
			else \
				color_enabled=1; \
			fi ;; \
	esac; \
	c_red=''; c_green=''; c_yellow=''; c_reset=''; \
	if [ "$$color_enabled" -eq 1 ]; then \
		c_red="$$(printf '\033[31m')"; \
		c_green="$$(printf '\033[32m')"; \
		c_yellow="$$(printf '\033[33m')"; \
		c_reset="$$(printf '\033[0m')"; \
	fi; \
	while IFS=$$(printf '\t') read -r op rel; do \
		[ -z "$$op" ] && continue; \
		abs='$(LOCAL_DIR)'/"$$rel"; \
		if [ -f "$$abs" ]; then \
			b="$$(wc -c < "$$abs" | tr -d '[:space:]')"; \
			size="$$(awk -v b="$$b" 'BEGIN { \
				if (b >= 1024*1024*1024) printf "%.2fG", b/(1024*1024*1024); \
				else if (b >= 1024*1024) printf "%.2fM", b/(1024*1024); \
				else if (b >= 1024) printf "%.2fK", b/1024; \
				else printf "%dB", b; \
			}')"; \
		else \
			size="N/A"; \
		fi; \
		note=""; \
		case "$$rel" in \
			log/*) \
				if grep -Fxq "$$rel" "$$tmp_log_conflict" && [ "$$op" != "-" ]; then \
					op="+"; \
					note=" [log-protect: keep local, add conflict copy]"; \
				fi ;; \
		esac; \
		if [ "$$color_enabled" -eq 1 ]; then \
			c="$$c_yellow"; \
			case "$$op" in \
				+) c="$$c_green" ;; \
				-) c="$$c_red" ;; \
			esac; \
			printf "%b%s%b %b%s%b (%s)%s\n" "$$c" "$$op" "$$c_reset" "$$c" "$$rel" "$$c_reset" "$$size" "$$note"; \
		else \
			echo "$$op $$rel ($$size)$$note"; \
		fi; \
	done < "$$tmp_changes"; \
	change_count="$$(wc -l < "$$tmp_changes" | tr -d '[:space:]')"; \
	awk '\
		function is_pull_file(tok) { \
			return (tok ~ /^[<>ch]f[^ ]{10}$$/); \
		} \
		{ \
			item = $$1; \
			if (!is_pull_file(item)) next; \
			path = $$0; sub(/^[^ ]+[[:space:]]+/, "", path); \
			print path; \
		} \
	' "$$tmp_log_all" | sort -u > "$$tmp_pull"; \
	files_changed=0; \
	total_bytes=0; \
	total_lines=0; \
	while IFS= read -r rel; do \
		[ -z "$$rel" ] && continue; \
		abs='$(LOCAL_DIR)'/"$$rel"; \
		if [ -f "$$abs" ]; then \
			b="$$(wc -c < "$$abs" | tr -d '[:space:]')"; \
			l="$$(wc -l < "$$abs" | tr -d '[:space:]')"; \
			total_bytes=$$((total_bytes + b)); \
			total_lines=$$((total_lines + l)); \
			files_changed=$$((files_changed + 1)); \
		fi; \
	done < "$$tmp_pull"; \
	human_size="$$(awk -v b="$$total_bytes" 'BEGIN { \
		if (b >= 1024*1024*1024) printf "%.2fG", b/(1024*1024*1024); \
		else if (b >= 1024*1024) printf "%.2fM", b/(1024*1024); \
		else if (b >= 1024) printf "%.2fK", b/1024; \
		else printf "%dB", b; \
	}')"; \
	rsync_file_count="$$(awk -F': ' '/^Number of regular files transferred:/ {print $$2; exit}' "$$tmp_log_all")"; \
	rsync_tx_size="$$(awk -F': ' '/^Total transferred file size:/ {print $$2; exit}' "$$tmp_log_all")"; \
	[ -z "$$rsync_file_count" ] && rsync_file_count=0; \
	[ -z "$$rsync_tx_size" ] && rsync_tx_size=0B; \
	echo "---- summary ----"; \
	echo "Changed entries: $$change_count"; \
	echo "Files to pull: $$files_changed"; \
	echo "Code volume (pulled files): $$human_size ($$total_bytes bytes)"; \
	echo "Code lines (pulled files): $$total_lines"; \
	echo "Rsync regular files transferred: $$rsync_file_count"; \
	echo "Rsync transferred file size: $$rsync_tx_size"; \
	if [ -s "$$tmp_list_log" ]; then \
		echo "Log files planned: $$log_plan_count"; \
		if [ "$$apply_mode" -eq 1 ]; then \
			echo "Log new files added: $$log_new_added"; \
			echo "Log conflict copies added: $$log_conflict_added"; \
		else \
			echo "Log new files to add: $$log_new_plan"; \
			echo "Log conflict copies to add: $$log_conflict_plan"; \
		fi; \
		echo "Log protect mode: existing local files under log/ are never overwritten/deleted."; \
	fi

pull-role:
	$(ensure_remote_sync_config)
	@set -e; \
	local_agent='$(LOCAL_DIR)/.agent'; \
	local_role="$$local_agent/ROLE.$(SYNC_HOST).md"; \
	remote_role='$(REMOTE_DIR)/.agent/ROLE.md'; \
	mkdir -p "$$local_agent"; \
	if ssh $(SSH_HOST) "test -f \"$$remote_role\""; then \
		rsync -azP -e ssh "$(SSH_HOST):$$remote_role" "$$local_role"; \
		echo "Pulled remote role to: $$local_role"; \
	else \
		echo "[WARN] Not found on remote: $$remote_role"; \
	fi

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
	' "$(ALIAS_FILE)"

config:
	$(ensure_remote_sync_config)
	@if [ ! -f "$(SSH_KEY)" ]; then \
		echo "SSH key not found: $(SSH_KEY)"; \
		exit 2; \
	fi
	ssh-copy-id -i $(SSH_KEY) $(SSH_HOST)
	@set -e; \
	rc_file="$(HOME)/.zshrc"; \
	begin="# >>> $(MODULE_NAME) alias init >>>"; \
	end="# <<< $(MODULE_NAME) alias init <<<"; \
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
	echo "Updated $$rc_file with $(MODULE_NAME) alias bootstrap."; \
	echo "To use mc/ms in this current shell now, run: eval \"\$$(make -s alias)\""
	@$(MAKE) role-local
	@$(MAKE) role-remote
	@$(MAKE) pull-role

role-local:
	@set -e; \
	mkdir -p "$(LOCAL_DIR)/.agent"; \
	role_kind="main-dev"; \
	if ! git -C "$(LOCAL_DIR)" rev-parse --is-inside-work-tree >/dev/null 2>&1; then role_kind="server-runtime"; fi; \
	ts="$$(date -u '+%Y-%m-%dT%H:%M:%SZ')"; \
	hn="$$(hostname -s 2>/dev/null || hostname)"; \
	usr="$$(whoami)"; \
	{ \
		echo "# ROLE"; \
		echo ""; \
		echo "- role_kind: $$role_kind"; \
		echo "- node: $$usr@$$hn"; \
		echo "- sync_host: $(SYNC_HOST)"; \
		echo "- remote_dir: $(REMOTE_DIR)"; \
		echo "- generated_at_utc: $$ts"; \
		echo ""; \
		echo "## Notes"; \
		echo "- main-dev: primary edit and git source of truth"; \
		echo "- server-runtime: execute workloads, keep data/models/cache local to server"; \
		echo ""; \
		echo "## Prompt"; \
		cat "$(ROLE_PROMPT_FILE)"; \
	} > "$(LOCAL_DIR)/.agent/ROLE.md"; \
	echo "Wrote local role: $(LOCAL_DIR)/.agent/ROLE.md ($$role_kind)"

role-remote:
	$(ensure_remote_sync_config)
	@set -e; \
	tmp_file="$$(mktemp)"; \
	ts="$$(date -u '+%Y-%m-%dT%H:%M:%SZ')"; \
	hn='$(SSH_HOST)'; \
	{ \
		echo "# ROLE"; \
		echo ""; \
		echo "- role_kind: server-runtime"; \
		echo "- node: $$hn"; \
		echo "- sync_host: $(SYNC_HOST)"; \
		echo "- remote_dir: $(REMOTE_DIR)"; \
		echo "- generated_at_utc: $$ts"; \
		echo ""; \
		echo "## Notes"; \
		echo "- This node can run workloads and may receive temporary edits."; \
		echo "- Pull selected paths back to main-dev via: make pull <list> y"; \
		echo ""; \
		echo "## Prompt"; \
		cat "$(ROLE_PROMPT_FILE)"; \
	} > "$$tmp_file"; \
	ssh $(SSH_HOST) "mkdir -p '$(REMOTE_DIR)/.agent'"; \
	rsync -azP -e ssh "$$tmp_file" "$(SSH_HOST):$(REMOTE_DIR)/.agent/ROLE.md"; \
	rm -f "$$tmp_file"; \
	echo "Wrote remote role: $(SSH_HOST):$(REMOTE_DIR)/.agent/ROLE.md"

role-show:
	@set -e; \
	echo "[local] $(LOCAL_DIR)/.agent/ROLE.md"; \
	if [ -f "$(LOCAL_DIR)/.agent/ROLE.md" ]; then sed -n '1,80p' "$(LOCAL_DIR)/.agent/ROLE.md"; else echo "missing"; fi; \
	echo ""; \
	echo "[remote] $(SSH_HOST):$(REMOTE_DIR)/.agent/ROLE.md"; \
	ssh $(SSH_HOST) "if [ -f '$(REMOTE_DIR)/.agent/ROLE.md' ]; then sed -n '1,80p' '$(REMOTE_DIR)/.agent/ROLE.md'; else echo 'missing'; fi"

bpe:
	$(ensure_remote_sync_config)
	@set -e; \
	mkdir -p "$(PROJECT_ROOT)/models"; \
	local_target="$(BPE_FILE)"; \
	local_src1="$(PROJECT_ROOT)/github/LLaVA-NeXT/llava/model/multimodal_encoder/dev_eva_clip/eva_clip/$(BPE_NAME)"; \
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
	echo "Cleaning Python cache files under $(PROJECT_ROOT)..."; \
	find "$(PROJECT_ROOT)" \
		\( -path "$(PROJECT_ROOT)/.git" -o -path "$(PROJECT_ROOT)/.venv" \) -prune -o \
		-type d \
		\( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name ".ipynb_checkpoints" \) \
		-prune -exec rm -rf {} +; \
	find "$(PROJECT_ROOT)" \
		\( -path "$(PROJECT_ROOT)/.git" -o -path "$(PROJECT_ROOT)/.venv" \) -prune -o \
		-type f \
		\( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) \
		-delete; \
	echo "Clean complete."
