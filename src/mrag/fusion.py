def fuse_reciprocal_rank(all_results: list[list[dict]], final_k: int = 5, rrf_k: int = 60) -> list[dict]:
    scores = {}
    payload = {}
    for result_list in all_results:
        for row in result_list:
            key = row["path"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + row["rank"])
            payload[key] = row
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:final_k]
    out = []
    for rank, (key, score) in enumerate(ranked, start=1):
        row = dict(payload[key])
        row["fusion_score"] = float(score)
        row["rank"] = rank
        out.append(row)
    return out


def fuse_score_sum(all_results: list[list[dict]], final_k: int = 5) -> list[dict]:
    scores = {}
    payload = {}
    for result_list in all_results:
        for row in result_list:
            key = row["path"]
            scores[key] = scores.get(key, 0.0) + float(row.get("score", 0.0))
            payload[key] = row
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:final_k]
    out = []
    for rank, (key, score) in enumerate(ranked, start=1):
        row = dict(payload[key])
        row["fusion_score"] = float(score)
        row["rank"] = rank
        out.append(row)
    return out


def fuse_voting(all_results: list[list[dict]], final_k: int = 5) -> list[dict]:
    votes = {}
    score_sum = {}
    payload = {}
    for result_list in all_results:
        for row in result_list:
            key = row["path"]
            votes[key] = votes.get(key, 0) + 1
            score_sum[key] = score_sum.get(key, 0.0) + float(row.get("score", 0.0))
            payload[key] = row
    ranked = sorted(votes.keys(), key=lambda k: (-votes[k], -score_sum[k]))[:final_k]
    out = []
    for rank, key in enumerate(ranked, start=1):
        row = dict(payload[key])
        row["fusion_votes"] = int(votes[key])
        row["fusion_score"] = float(score_sum[key])
        row["rank"] = rank
        out.append(row)
    return out
