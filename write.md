Filters and Transformations that worked to best increase the accuracy of our classifier we as follows:
1. Filters:
    1. is_empty: Here we filter out empty words or empty strings. This is extremely crucial because we don't want a correlation forming between these.
    2. is_small: Words smaller than 3 characters tend to be less important. Words like "the", "fan", etc are likely to occur in both positive and negative reviews. Therefore, filtering them out helps us get rid of noise and increases our accuracy.
    3. is_grammer: Removing all grammar from our tokens helps us filter out all punctuation marks and thus helps us get rid of noise and increases accuracy.
    4. is_allowed_part_of_speech: In our case I noticed that a lot of the text is actually superflous and verbs aren't helping out. So after applying that filter of only allowing nouns, adjectives, adverbs and verbs, I noticed an increase in accuracy.

2. Tranforms:
    1. For transformations tokenizing the words helped me increase my accuracy. This is because tokenizing the words helped me further get rid of all the punctuation marks and thus helped me get rid of noise.

The regex parser actually parsed most of the sentences as expected and was fairly accurate.
However, there were minor misidentifications that could lead to bigger problems in the future.
For instance, in the second sentence, the 's' from microsoft's  was tagged as a Verb Past Tense (VBD).
This is incorrect because the 's' actually indicates a possessive noun (NNP). This goes to show that the regex parser, even though not perfect, is fairly accurate.


The chunker identifying named entities was also fairly accurate. However, there were somesomewhat major misidentifications here as well. One of them was that the word "Gates" was tagged as a GPE (Geopolitical Entity) instead of a PERSON. This is a major misidentification because the word "Gates" is referring to Bill Gates, a person, and not a geopolitical entity. this is especially odd since the bhinker later on identifies both 'BILL' and 'GATES' as a person. That leads to me to believe the grammar in the sentence led the chunker down the wrong path earlier. The chunker also reffered to Bitcoin as a person in one of the sentences, which is incorrect. Makes me wonder how the chunker identifies names or persons. Lastly, other than identifying Energy as GPE, the chunker performed faily well.
