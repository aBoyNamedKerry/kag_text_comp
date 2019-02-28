library(tidytext)
library(tidyverse)
library(magrittr)


train_tokenized<- read.csv("./data/train_tokenized.csv", header = TRUE,
                           stringsAsFactors = FALSE)

test_tokenized<- read.csv("./data/test.csv", header = TRUE,
                          stringsAsFactors = FALSE)

#look at data which has been tokenized
head(train_tokenized)

#look at proportion of labelled data
prop.table(table(train_tokenized$is_duplicate))

#check tokenization function out

train_tokenized_1 <- train_tokenized %>% 
 # mutate(id = row_number()) %>%
  unnest_tokens(words_1, question1) 

train_tokenized_2 <- train_tokenized %>% unnest_tokens(words_2, question2)


#get sentiments

#note some of the question numbers appear in both columns
#for overall sentiment use Bing algorithm although less words covered
tokenized_sentiment1 <- train_tokenized_1  %>% inner_join(get_sentiments("bing"),
                                           by = c("words_1" = "word")) %>%
  count(X, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  mutate(overall_sentiment = ifelse(sentiment>0,"postive",
                                    ifelse(sentiment == 0, "neutral", "negative")))



tokenized_sentiment2 <- train_tokenized_2  %>% inner_join(get_sentiments("bing"),
                                                          by = c("words_2" = "word")) %>%
  count(X, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  mutate(overall_sentiment = ifelse(sentiment>0,"postive",
                                    ifelse(sentiment == 0, "neutral", "negative")))


## join the data --------------

#join first sentiment
train_tokenized <- tokenized_sentiment1 %>%
  select(X, sentiment_q1 = sentiment, overall_sentiment_q1 = overall_sentiment) %>%
  left_join(train_tokenized,.)

#join second
train_tokenized <- tokenized_sentiment2 %>%
  select(X, sentiment_q2 = sentiment, overall_sentiment_q2 = overall_sentiment) %>%
  left_join(train_tokenized,.)

#Mutate columns to set no sentiment to zero and then to neutral
# this is to remove NAs and produce a sentiment score

train_tokenized %<>% 
  mutate(sentiment_q1 = ifelse(is.na(sentiment_q1), 0, sentiment_q1),
         sentiment_q2 = ifelse(is.na(sentiment_q2), 0, sentiment_q2)) %>%
  mutate(overall_sentiment_q1 = ifelse(sentiment_q1 == 0, "neutral", overall_sentiment_q1),
         overall_sentiment_q2 = ifelse(sentiment_q2 == 0, "neutral", overall_sentiment_q2))

# derive cobined sentiment
#this is speratd out for explination purposes as could go above
#absolutes used just to show gaps and remove negativ and positive numbers
train_tokenized %<>%
  mutate(same_sentiment = ifelse(overall_sentiment_q1 == overall_sentiment_q2, 1,0),
         sentiment_difference = abs(sentiment_q1 - sentiment_q2))


## Write the data out without id column as this is not used by Python algorithm
train_tokenized %>% select(same_sentiment, sentiment_difference) %>%
  write.csv(., "./features_train/sentiment_feature_selection.csv",
            row.names = FALSE)

#check if sentiment differences is as expected
round(prop.table(table(train_tokenized$same_sentiment, 
      train_tokenized$sentiment_difference),1)*100,2)


## create feature for test ==============================

#filter to the n umber needed for submission 
test_tokenized<- test_tokenized[0:2345796,]


#create the nested word splits for doing the sentiment analysis
test_tokenized_1 <- test_tokenized %>% 
  unnest_tokens(words_1, question1) 

test_tokenized_2 <- test_tokenized %>% unnest_tokens(words_2, question2)


#join to sentiment analysis list
tokenized_sentiment1 <- test_tokenized_1  %>% inner_join(get_sentiments("bing"),
                                                          by = c("words_1" = "word")) %>%
  count(test_id, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  mutate(overall_sentiment = ifelse(sentiment>0,"postive",
                                    ifelse(sentiment == 0, "neutral", "negative")))



tokenized_sentiment2 <- test_tokenized_2  %>% inner_join(get_sentiments("bing"),
                                                          by = c("words_2" = "word")) %>%
  count(test_id, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative) %>%
  mutate(overall_sentiment = ifelse(sentiment>0,"postive",
                                    ifelse(sentiment == 0, "neutral", "negative")))


#remove the data here as these are big data set

rm(test_tokenized_1, test_tokenized_2)
gc()

## join the data --------------

#join first sentiment
test_tokenized <- tokenized_sentiment1 %>%
  select(test_id, sentiment_q1 = sentiment, overall_sentiment_q1 = overall_sentiment) %>%
  left_join(test_tokenized,.)

#join second
test_tokenized <- tokenized_sentiment2 %>%
  select(test_id, sentiment_q2 = sentiment, overall_sentiment_q2 = overall_sentiment) %>%
  left_join(test_tokenized,.)





## Testing code =====================
test <- train_tokenized_1  %>% left_join(get_sentiments("nrc"),
                                                          by = c("words_1" = "word")) %>%
  count(qid1, sentiment)
