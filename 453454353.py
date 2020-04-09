import re


s="adfkasdjklfjdslf???"
s = re.sub(r"([.!?])", r" \1", s)
s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
print(s)
