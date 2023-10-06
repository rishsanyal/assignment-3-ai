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

## TODO: Add The writeup

**(10 points)** Compare the performance of your classifier with and without feature selection on the movie_reviews data using five-fold cross-validation.
(you can either use sklearn, or your code from assignment 2.) Then compare this to the [NLTK implementation of Naive Bayes](https://www.nltk.org/book/ch06.html) using five-fold cross-validation.

In your writeup, add a paragraph that describes the most effective set of filters and transforms, and the accuracy you were able to achieve using these.

Question 4. Named Entity Extraction. An alternative approach to textual analysis is to explicitly look at the structure of the sentences using a parser. NLTK has a set of tools included that allow us to do this. The file chunking.py shows how it works with a simple example.

**(10 points)** To begin, find at least five English-language sentences from news sources to use as input. Ones that talk about specific people, companies or places are best. Run them through the RegexpParser shown in chunking.py. This will extract noun phrases. Add a paragraph to your writeup indicating which phrases were correctly extracted, and which were not. Do you see any pattern to the phrases that were missed?

**(10 points)** Next, run the named entity chunker over your same examples. This will extract named entities, which are noun phrases that refer to specific people, places or things. Add a paragraph to your writeup indicating which noun phrases were correctly extracted. Do you see a pattern to the phrases that were missed?

## TODO: Write the write up

**(686 Students only)** Please read [this article](https://thegradient.pub/machine-learning-wont-solve-the-natural-language-understanding-challenge/). and answer the following questions:

According to the author, what is the difference between natural language processing and natural language understanding?

In the article, the author is trying to explain that NLU isn't a micro-level data problem but isntead is a macro-level problem. NLP focuses on actions like tokenization, stemming, lemmazation, etc. NLU focuses on the meaning of the language and has a higher, bird's eye level view.Additionally, while NLP operates on the PAC paradigm and NLU doesn't, the author expands on how there is more room for misinterpreting inputs by NLU but it can be judged and corrected. This NLU focuses on the context and operates beyond the mere assocations drawn between characters and words.

Why does the author feel that data-driven approaches are not suitable for NLU?

According to the author, while NLP tasks are subjective, NLU tasks are not. NLP tasks operate on Probably Approximate Correct paradigm, whereas for NLU a complete understaing of the context or language is required, which actually can be judged by the speaker or input. This is also expanded upon when the author talks about how two conversing parties need to have a shared knowledge of the background of the conversation. Data driven approaches can't always have context and can't always be judged by the speaker. For example, with data-driven approaches it's easier to misinterpret the meaning of a sentence like "I'm going to bet my life" here the speaker is not going to bet his life, but rather is going to bet a lot of money, but a data-driven approach would not be able to understand this. This example also shows how the author believes that data-driven approaches don't process the understanding of the sentence or context and need to process "multi-dimensional spaces that cover an entire dataset".

If we accept the author's premise, what sorts of tasks are then best suited for data-driven approaches to NLP?

The author believes, smaller tasks related to text-processing would be ideal for data-driven approaches for NLP. Tasks like clustering, topic extraction and automatic tagging would be best for data-driven approaches. These tasks are limited in scope, depend solely on the input (don't need context) and don't necessarily need a larger understading of language or phrases.