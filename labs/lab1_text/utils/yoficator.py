import os
import re
import codecs

class CustomYoficator:
    def __init__(self, dictionary_path=None):
        if dictionary_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dictionary_path = os.path.join(current_dir, "yo.dat")
            
        self.dictionary = {}
        self.splitter = re.compile(r'(\s+|\w+|\W+|\S+)', re.UNICODE)
        if os.path.exists(dictionary_path):
            self._load_dictionary(dictionary_path)

    def _load_dictionary(self, path):
        with codecs.open(path, "r", "utf-8") as f:
            for line in f:
                cline = line.rstrip('\n')
                if "(" in cline:
                    bline, sline = cline.split("(")
                    sline = re.sub(r'\)', '', sline)
                else:
                    bline = cline
                    sline = ""
                
                if "|" in sline:
                    ssline = sline.split("|")
                    for ss in ssline:
                        value = bline + ss
                        value = re.sub(r'\*', '', value)
                        key = re.sub(r'ё', 'е', value)
                        self.dictionary[key] = value
                else:
                    value = bline
                    value = re.sub(r'\*', '', value)
                    key = re.sub(r'ё', 'е', value)
                    self.dictionary[key] = value

    def check_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return 0, text
        
        tokens = self.splitter.findall(text)
        has_fixes = False
        fixed_tokens = []
        
        for token in tokens:
            if token in self.dictionary:
                fixed_tokens.append(self.dictionary[token])
                has_fixes = True
            elif token.lower() in self.dictionary:
                fixed_val = self.dictionary[token.lower()]
                if token.istitle():
                    fixed_val = fixed_val.title()
                elif token.isupper():
                    fixed_val = fixed_val.upper()
                fixed_tokens.append(fixed_val)
                has_fixes = True
            else:
                fixed_tokens.append(token)
                
        return (1 if has_fixes else 0), "".join(fixed_tokens)