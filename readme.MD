### Assignment 3: Introduction to Natural Language and Probability

#### Due: Friday, October 6 at 11:59pm.

To submit: Please check all code into your GitHub repo, along with a PDF containing the answers to the written questions. Also, please provide a submission.py that demonstrates all of your code.

**(25 points) 1.Intro to Probability questions.**

Please watch the Intro to Probability video and work through the questions presented in the video.
Please include a PDF in your repo with the answers to these questions.

**(25 points) 2. Natural Language.* 

Please work through the "Intro to NLTK" document on the Canvas website. Include a Python file called nltk_into.py
in your repo containing the answers to these tasks.


**Part 3: Document classification with Naive Bayes**

In this section, you'll implement your first "real" machine learning algorithm: Naive Bayes. 
I've given you a lot of the basic machinery. We'll test out our algorithm on NLTK's movie_reviews dataset.

(10 points) nb.py contains most of the code for building a Naive Bayes classifier. You just need to complete classify, using log-likelihood to return a dictionary mapping categories (pos, neg) to their likelihoods. You will probably also want to add some functionality to the main.

(10 points) Next, we need to think about feature selection. This is a very important problem when we're dealing with text; if we can identify and properly weight the important features, we can greatly improve performance. I've provided some initial filters and transformations in filters.py, along with some code to allow you to dynamically reconfigure them.

Filters will return a token if it matches their filter, and False otherwise. For example, alphabetic filters words that contain only alphabetic characters, and stopword returns a token if it is not a stopword.

Transforms will take a token as input and return a new, transformed token. For example, trim() removes whitespace, and lowercase converts to lower case.

Look at the movie_reviews data - what do you see in there that might be helpful? Use this to create at least four additional filters or transforms that you think will improve the performance of Naive Bayes.

**(10 points)** Compare the performance of your classifier with and without feature selection on the movie_reviews data using five-fold cross-validation.
(you can either use sklearn, or your code from assignment 2.) Then compare this to the [NLTK implementation of Naive Bayes](https://www.nltk.org/book/ch06.html) using five-fold cross-validation. 

In your writeup, add a paragraph that describes the most effective set of filters and transforms, and the accuracy you were able to achieve using these.

Question 4. Named Entity Extraction. An alternative approach to textual analysis is to explicitly look at the structure of the sentences using a parser. NLTK has a set of tools included that allow us to do this. The file chunking.py shows how it works with a simple example.

**(10 points)** To begin, find at least five English-language sentences from news sources to use as input. Ones that talk about specific people, companies or places are best. Run them through the RegexpParser shown in chunking.py. This will extract noun phrases. Add a paragraph to your writeup indicating which phrases were correctly extracted, and which were not. Do you see any pattern to the phrases that were missed?

**(10 points)** Next, run the named entity chunker over your same examples. This will extract named entities, which are noun phrases that refer to specific people, places or things. Add a paragraph to your writeup indicating which noun phrases were correctly extracted. Do you see a pattern to the phrases that were missed?

**(686 Students only)** Please read [this article](https://thegradient.pub/machine-learning-wont-solve-the-natural-language-understanding-challenge/). and answer the following questions:

According to the author, what is the difference between natural language processing and natural language understanding?

Why does the author feel that data-driven approaches are not suitable for NLU?

If we accept the author's premise, what sorts of tasks are then best suited for data-driven approaches to NLP?
