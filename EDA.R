library(tidytext)
library(tidyverse)
library(magrittr)


train_tokenized<- read.csv("./data/train_tokenized.csv", header = TRUE,
                           stringsAsFactors = FALSE)

#look at data which has been tokenized
head(train_tokenized)

#look at proportion of labelled data
prop.table(table(train_tokenized$is_duplicate))

#check tokenization function out

train_tokenized_1 <- train_tokenized %>% unnest_tokens(words_1, question1)

train_tokenized_2 <- train_tokenized %>% unnest_tokens(words_2, question2)


#get sentiments