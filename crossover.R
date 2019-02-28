## Code for building crossover features and 


#read in data
data <- read.csv("./data/test.csv",header=TRUE,
                 stringsAsFactors = FALSE)
#test
data<- data[1:2345796,]

#load libraries required
library("wordnet")
library("dplyr")
library("tidytext")
library("tm")
library("textstem")
library("tidyr")
library(magrittr)

data$question1 <- tolower(data$question1)
data$question2 <- tolower(data$question2)

data$question1 <- removePunctuation(data$question1)
data$question2 <- removePunctuation(data$question2)


Output<-as.data.frame(data$id)

#crossover calculated

countOfSame <- function(s)
{
  merged <- merge(unlist(strsplit(s[1]," ")),unlist(strsplit(s[2]," ")))
  return(sum(apply(merged[!duplicated(merged),],1,function(x) {ifelse(toupper(x[1]) == toupper(x[2]),TRUE,FALSE)}))/mean(length(unlist(strsplit(s[1]," "))),length(unlist(strsplit(s[2]," ")))))
}



countOfSame_2 <- function(data)
{
  df <- tibble(
    x = data$X,
    y = data$question1,
    z = data$question2
  )
  
  test<-df %>%
    transform(y = strsplit(y, " ")) %>%
    unnest(y) %>%
    transform(z = strsplit(z, " ")) %>%
    unnest(z) 
  
  test$score[test$y==test$x]<-1
  test$score[!test$y==test$x]<-0
  
  test<-test %>% group_by(X) %>% summarise(len=mean(length(unique(y)),length(unique(x))),score=sum(score)/len)
  
  test$score                                       
  
}



#data_questions<-data[,5:6]
#Output$score<-apply(data_questions,1,countOfSame)


#Output$Crossover<-countOfSame_2(data)


#Stopwords remoevd and cross over calculated

data_stopwords<-data

corpus_q1<-Corpus(VectorSource(data_stopwords$question1))
corpus_q2<-Corpus(VectorSource(data_stopwords$question2))

q1_tm<-tm_map(corpus_q1, removeWords, stopwords('english'))
q2_tm<-tm_map(corpus_q2, removeWords, stopwords('english'))

data_stopwords$question1 <- unlist(q1_tm)[1:2345796]

data_stopwords$question2 <-  unlist(q2_tm)[1:2345796]

remove(corpus_q1,corpus_q2,q1_tm,q2_tm);gc()

#data_stopwords$question1<-as.character(data_stopwords$question1)
#data_stopwords$question2<-as.character(data_stopwords$question2)


data_questions_stopwords<-data_stopwords[,2:3]
#data$score_stopwords_removed<-apply(data_questions_stopwords,1,countOfSame)

Output$Crossover_stopwords<-apply(data_questions_stopwords,1,countOfSame)


#export to excel

#write.csv(Output,file="T:/Output.csv")

rm(data, data_questions, data_questions_stopwords)
gc()

#Stopwords remeoved and lemmatized

data_lem<-data_stopwords
#rm(data_stopwords) ; gc()

#lem_q1<-data$question1
#lem_q2<-data_lem$question2

data_lem$question1<-lemmatize_strings(data$question1)
data_lem$question2<-lemmatize_strings(data$question2)

data_questions_stopwords<-data_lem[,2:3]

Output$Crossover_stopwords_lim<-apply(data_questions_stopwords,1,countOfSame)



## Add word count to test

data %<>% mutate(q1_words_tot = sapply(strsplit(question1, " "), length),
                 q2_words_tot = sapply(strsplit(question2, " "), length))


#write out the data to test
data %>% select(q1_words_tot, q2_words_tot) %>%
  write.csv(., "./features_test/word_count_train.csv")


## Test data
 test_data <- head(data)
 
 test_data %>% mutate(q1_words_tot = stringr::str_count(data$question1, "//w+"))
 
 
 sapply(strsplit(test_data$question1, " "), length)
 
##test the lemmatization

lem_q1_test <- lem_q1[1:10000]

lem_sting_test<- lemmatize_strings(lem_q1_test)



head(lem_q1)
