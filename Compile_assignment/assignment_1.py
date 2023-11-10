import re

# Define regular expressions for tokens
patterns = [
    (r'int|float|double|char|if|then|else|switch|case|break|continue|while|do|for', 1),
    (r'[a-zA-Z_]\w*', 10),
    (r'\d+', 11),
    (r'[-+*/<<===!=><=>;()?:,]', {
        '=': 17, '+': 13, '-': 14, '*': 15, '/': 16, '<': 20, '<=': 21, '==': 22, '!=': 23,
        '>': 24, '>=': 25, ';': 26, '(': 27, ')': 28, '?': 29, ':': 30, ',': 31
    }),
]

'''
Function Description:
    This function checks whether a given input string is a valid identifier based on a regular expression pattern.

Regular Expression Explanation:
    - ^: Indicates matching from the beginning of the string.
    - [a-zA-Z_]: Matches a single letter (uppercase or lowercase) or an underscore.
    - \w*: Matches zero or more word characters (letters, digits, or underscores).
    - $: Denotes matching up to the end of the string.

Overall, the regular expression pattern means:
    - Start matching from the beginning of the string.
    - Match a single letter (uppercase or lowercase) or an underscore.
    - Then, match zero or more word characters.
    - Finally, ensure the match reaches the end of the string.

Parameters:
    s (str): The input string to be checked.

Returns:
    bool: True if the input string is a valid identifier, False otherwise.
'''
def isIdentifier(s):
    return not isKeywords(s) and re.match(r'^[a-zA-Z_]\w*$', s)


'''
Function Description:
    This function checks whether a given input string is a valid keyword based on a regular expression pattern.
Keyword List:
    The function checks if the input string exists in a predefined list of keywords, including:
    - "main"
    - "int"
    - "float"
    - "double"
    - "char"
    - "if"
    - "then"
    - "else"
    - "switch"
    - "case"
    - "break"
    - "continue"
    - "while"
    - "do"
    - "for"
Parameters:
    s (str): The input string to be checked.
Returns:
    bool: True if the input string is a valid keyword, False otherwise.
'''
def isKeywords(s):
    keywords = ["main", "int", "float", "double", "char", "if", "then", "else", "switch", "case", "break", "continue", "while", "do", "for"]
    return s in keywords

'''
Function Description:
    This function checks whether a given input string is a valid number based on a regular expression pattern.
Parameters:
    s (str): The input string to be checked.
Returns:
    bool: True if the input string is a valid number, False otherwise.
'''
def isDigit(s):
    return re.match(r'^\d+$', s)

'''
Function Description:
    This function checks whether a given input string is a valid operator based on a regular expression pattern.
Keyword List:
    The function checks if the input string exists in a predefined list of keywords, including:
        "=", "+", "-", "*", "/", "<", "<=", "==", "!=", ">", ">=", ";", "(", ")", "?", ":", ","
Parameters:
    s (str): The input string to be checked.
Returns:
    bool: True if the input string is a valid operator, False otherwise.
'''
def isOperator(s):
    operators = ["=", "+", "-", "*", "/", "<", "<=", "==", "!=", ">", ">=", ";", "(", ")", "?", ":", ","]
    return s in operators

def tokenize_and_categorize(input_data):
    tokens = []
    pattern_str = '|'.join(f'({p[0]})' for p in patterns)
    for match in re.finditer(pattern_str, input_data):
        for i, p in enumerate(patterns):
            if match.group(i + 1):
                if p[1] == 11:
                    tokens.append((p[1], int(match.group(i + 1))))
                else:
                    tokens.append((p[1], match.group(i + 1)))
    return tokens


def result(token):
    s = token[1] if isinstance(token[1], str) else str(token[1])
    keyMap = {
        "int": "1",
        "float": "2",
        "if": "3",
        "else": "4",
        "switch": "5",
        "while": "6",
        "for": "7",
    }

    opeMap = {
        "=": "(17,=)",
        "<": "(less than--20,<)",
        "<=": "(less than or equal to--21,<=)",
        "==": "(equal operator--22,==)",
        "!=": "(not equal operator--23,!=)",
        ">": "(bigger than--24,>)",
        ">=": "(bigger than or equal to--25,>=)",
        ";": "(26,;)",
        "+": "(13,+)",
        "(": "(left bracket--27,()",
        "-": "(minus--14,-)",
        ")": "(right bracket--28,))",
        ">": "(bigger than operator--24,>)",
        "*": "(star--15,*)",
        "?": "(question--29,?)",
        "/": "(divide--16,/)",
        ":": "(30,:)",
        ",": "(31,,)"
    }

    if isIdentifier(s):
        return f"(10,{s})"
    elif isKeywords(s):
        return f"({keyMap.get(s, 'Unknown')},{s})"
    elif isDigit(s):
        return f"(11,{s})"
    elif isOperator(s):
        return opeMap.get(s, "Error")
    else:
        return "Error"

input_file = 'TestData.txt'
output_file = 'Result.txt'

with open(input_file, 'r') as input_data:
    input_content = input_data.read()

tokens = tokenize_and_categorize(input_content)

with open(output_file, 'w') as output:
    output.write("-----Original data-----:\n")
    output.write(input_content)
    output.write("\n-----Handled data-----:\n")

    for token in tokens:
        s = result(token)
        print(s)
        output.write(s + '\n')
