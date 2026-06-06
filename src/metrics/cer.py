def levenshtein_distance(pred, target):
    dp = [[0] * (len(target) + 1) for _ in range(len(pred) + 1)]

    for i in range(len(pred) + 1):
        dp[i][0] = i

    for j in range(len(target) + 1):
        dp[0][j] = j

    for i in range(1, len(pred) + 1):
        for j in range(1, len(target) + 1):
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[-1][-1]

def character_error_rate(pred, target):
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(pred, target) / len(target)

def average_cer(preds, targets):
    if len(targets) == 0:
        return 0.0

    total = 0.0
    for pred, target in zip(preds, targets):
        total += character_error_rate(pred, target)
    return total / len(targets)