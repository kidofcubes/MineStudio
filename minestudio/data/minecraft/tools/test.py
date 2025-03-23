import re
CONTRACTOR_PATTERN = r"^(.*?)-(\d+)$"
file_name = "woozy-ruby-ostrich-f153ac423f61-20220420-152230"
match = re.match(CONTRACTOR_PATTERN,file_name )
if match:
    eps, ord = match.groups()
else:
    eps, ord = file_name, "0"
    
print(eps,ord)
