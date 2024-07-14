#### Utils: Delete unnecessary infos in the log.

def clean_log(log):
    filter_patterns = [
        "[LightGBM] [Info]",
        "You can set `force_col_wise=true` to remove the overhead."
    ]
    logs = log.split('\n')
    new_log = ""
    for log in logs:
        flag = False
        for filter in filter_patterns:
            if filter in log:
                flag = True
        if flag: continue
        new_log += log
        new_log += '\n'
    return new_log
        