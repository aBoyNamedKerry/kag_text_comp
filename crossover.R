

data <- read.csv("P:/data.csv",header=TRUE)

library("wordnet")
library("dplyr")
library("tidytext")
library("tm")
library("textstem")
library("tidyr")

data$question1 <- tolower(data$question1)
data$question2 <- tolower(data$question2)

data$question1<-as.character(data$question1)
data$question2<-as.character(data$question2)

data$question1 <- removePunctuation(data$question1)
data$question2 <- removePunctuation(data$question2)


Output<-as.data.frame(data$X)

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



data_questions<-data[,5:6]
Output$score<-apply(data_questions,1,countOfSame)


#Output$Crossover<-countOfSame_2(data)


#Stopwords remoevd and cross over calculated

data_stopwords<-data

library(tm)

corpus_q1<-Corpus(VectorSource(data_stopwords$question1))
corpus_q2<-Corpus(VectorSource(data_stopwords$question2))

q1_tm<-tm_map(corpus_q1, removeWords, stopwords('english'))
q2_tm<-tm_map(corpus_q2, removeWords, stopwords('english'))

data_stopwords$question1 <- data.frame(text=sapply(q1_tm, identity), 
                                       stringsAsFactors=F)

data_stopwords$question2 <-  data.frame(text=sapply(q2_tm, identity), 
                                        stringsAsFactors=F)

remove(corpus_q1,corpus_q2,q1_tm,q2_tm)

data_stopwords$question1<-as.character(data_stopwords$question1)
data_stopwords$question2<-as.character(data_stopwords$question2)


data_questions_stopwords<-data_stopwords[,5:6]
#data$score_stopwords_removed<-apply(data_questions_stopwords,1,countOfSame)

Output$Crossover_stopwords<-apply(data_questions_stopwords,1,countOfSame)


#export to excel

write.csv(Output,file="T:/Output.csv")


#Stopwords remeoved and lemmatized

library(tidytext)

data_lem<-data_stopwords

lem_q1<-as.vector(data_lem$question1)
lem_q2<-as.vector(data_lem$question2)

data_lem$question1<-lemmatize_strings(lem_q1)
data_lem$question2<-lemmatize_strings(lem_q2)

Output$Crossover_stopwords<-apply(data_lem,1,countOfSame)



Output$Crossover_stopwords<-countOfSame_2(data_stopwords)
