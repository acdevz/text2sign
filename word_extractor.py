import re
import os

files = os.listdir('./static/SignFiles');
word=re.compile(r'[^\/]+(?=\.)');
with open("words.txt",'w') as words_file:
	for f in files:
		if(word.match(f)):
			words_file.write(word.match(f).group())
			words_file.write("\n")
