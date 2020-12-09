from wiki_dump_reader import Cleaner, iterate
import csv
import os
from tqdm import tqdm

PATH_WIKI_XML = "/media/darg1/Data/datasets/enwiki"
PATH_CSV = "/media/darg1/Data/Projects/Concreteness/data"
FILENAME_WIKI = "enwiki-20200601-pages-articles.xml"
FILENAME_ARTICLES = "articles.csv"
FILENAME_REDIRECT = "articles_redirect.csv"
FILENAME_TEMPLATE = "articles_template.csv"




def main():
	pathWikiXML = os.path.join(PATH_WIKI_XML, FILENAME_WIKI)
	pathArticles = os.path.join(PATH_CSV, FILENAME_ARTICLES)
	pathArticlesRedirect = os.path.join(PATH_CSV, FILENAME_REDIRECT)
	pathTemplateRedirect = os.path.join(PATH_CSV, FILENAME_TEMPLATE)

	templateCount = 0
	articleCount = 0
	totalCount = 0
	redirectCount = 0

	with open(pathArticles, 'w') as output_file:
		cw = csv.writer(output_file, delimiter='\t')
		cw.writerow(['Title', 'Text'])

	cleaner = Cleaner()
	for title, text in tqdm(iterate(pathWikiXML)):
		totalCount += 1
		text = cleaner.clean_text(text)
		#cleaned_text, links = cleaner.build_links(text)

		if text.startswith("REDIRECT"):
			redirectCount += 1
		elif text.startswith("TEMPLATE"):
			templateCount += 1
		else:
			articleCount += 1
			with open(pathArticles, 'a') as output_file:
				cw = csv.writer(output_file, delimiter='\t')
				cw.writerow([title, text])

	print("Total pages: {:,}".format(totalCount))
	print("Template pages: {:,}".format(templateCount))
	print("Article pages: {:,}".format(articleCount))
	print("Redirect pages: {:,}".format(redirectCount))






if __name__ == "__main__":
	main()




