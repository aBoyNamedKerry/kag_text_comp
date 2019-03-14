#load libraries required

library(tidyverse)
library(tidytext)
library(topicmodels)

#read in the tokenised data
train_token <- read_csv("data/train_tokenized/train_tokenized.csv")

#create new columns for all of the tokenised data and ids, select only these columns in the df
#create a df for questions1
question_1 <- train_token %>%
  select(question = question1, id = qid1) 

#create a df for questions2
question_2 <- train_token %>%
  select(question = question2, id = qid2)

#create a df for all questions
all_questions <- rbind(question_1, question_2) 

# split into words
by_question_word <- all_questions %>%
  unnest_tokens(word, question) 

# find document-word counts & remove stopwords
word_counts <- by_question_word %>%
  anti_join(stop_words) %>%
  count(id, word, sort = TRUE) %>%
  ungroup()

# change dataframe to a Document Term Matrix
questions_dtm <- word_counts %>%
  cast_dtm(id, word, n)

# clean the environment
rm(question_1)
rm(question_2)
rm(train_token)
rm(all_questions)
rm(by_question_word)
gc()

# create model paraments for topics in the data
k = 10
seed = sample(1:1000000, 1)

# A LDA_VEM topic model with 10 topics.
questions_lda <- LDA(questions_dtm, k = k, control = list(seed = seed))
questions_lda

# question topics and their associated probabilities of being part of these topics
question_topics <- tidy(questions_lda, matrix = "beta")
question_topics

# examine the top 5 terms from each topic, which will hopefully show us topic differences
top_terms <- question_topics %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms

# visualise the top 5 words in these topics 
top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# get the per-question probabilities to matcht them back to the questions
# an estimated proportion of words from that document that are generated from that topic
topics_gamma <- tidy(questions_lda, matrix = "gamma")

# regroup by question
regroup_questions <- topics_gamma %>%
  group_by(document, topic) %>%
  summarise(gamma)

# reduce the dataset back to individual question ids with the highest weighted topic value only
max_topic_question <- regroup_questions %>%
  filter(gamma == max(gamma)) %>%
  arrange(document, topic) %>%
  rename(id = document) %>%
  ungroup()

# save to a csv
write.csv(max_topic_question, file = "maxtopicquestion.csv")

# get rid of some environment variable
rm(k)
rm(seed)
rm(question_topics)
rm(top_terms)
rm(regroup_questions)
rm(topics_gamma)
gc()

# join the data by id back to the original dataset id1
  
join_1 <- max_topic_question %>%
  mutate(id = as.numeric(id)) %>%
  left_join(train_token, ., by = c("qid1" = "id")) 

# join the data by id back to the original dataset with id2, giving us our final set
questions_weighted <- max_topic_question %>%
  mutate(id = as.numeric(id)) %>%
  left_join(join_1, ., by = c("qid2" = "id")) %>%
  select(question1, `topic.x`, `gamma.x`, question2, `topic.y`, `gamma.y`) %>%
  mutate(topic.x = ifelse(is.na(topic.x),0,topic.x), 
         gamma.x = ifelse(is.na(gamma.x),0,gamma.x),
         topic.y = ifelse(is.na(topic.y),0,topic.y),
         gamma.y = ifelse(is.na(gamma.y),0,gamma.y)) %>%
  mutate(same_topic = ifelse(topic.x == topic.y, 1, 0))

write.csv(questions_weighted, file = "questions_weight.csv")

# committ to github



