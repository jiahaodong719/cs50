import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dic = corpus.copy()
    for key in dic:
        dic[key] = (1-damping_factor)/len(corpus)
    link = corpus[page]
    if link:
        for i in link:
            dic[i] = dic[i] + damping_factor/len(link)
    else:
        for key in dic:
            dic[key] = 1/len(corpus)
    return dic


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample = []
    pages = list(corpus.keys())
    state = random.choices(pages, k=1)[0]
    for _ in range(n):
        tra = transition_model(corpus,state,damping_factor)
        state = random.choices(list(tra.keys()), weights = list(tra.values()), k=1)[0]
        sample.append(state)
    dic = corpus.copy()
    for key in dic:
        dic[key] = sample.count(key)/len(sample)
    return dic



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    page_link = dict([(k,[]) for k in list(corpus.keys())])
    dic = corpus.copy()
    for key in dic:
        for page in corpus:
            if key in corpus[page]:
                page_link[key].append(page)        
    for key in dic:
        dic[key] = 1/N
    diff = 1
    while diff>=0.001:
        copy_dic = dic.copy()
        for key in dic:
            dic[key] = (1-damping_factor)/N + damping_factor*\
                sum([copy_dic[page]/len(corpus[page]) for page in page_link[key]])
        diff = max([abs(i-j) for i,j in zip(dic.values(),copy_dic.values())])
    return dic
if __name__ == "__main__":
    main()
