ENDC = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

class level:
    info = "\033[94m[DEBUG]" + ENDC
    success = "\033[92m[ + ]" + ENDC
    warn = BOLD + "\033[93m[WARN]" + ENDC
    fail = BOLD + "\033[91m[FAIL]" + ENDC


def debug( text:str, lvl: str=level.info ):
    print( lvl + " " + text )
