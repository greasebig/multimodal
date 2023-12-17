# 创建一个字典
def minPathSum(self, path):
    """
    :type path: List[int]
    :rtype: int
    """
    if not path:
        return 0
    m = len(path)
    n = len(path[0])
    dp = [[0 for _ in range(n)] for _ in range(m)]
    dp[0][0] = path[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + path[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + path[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + path[i][j]
            



