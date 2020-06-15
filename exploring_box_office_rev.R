### disable scientific notation
options(scipen=999) 

# load and/or installs required packages
required_packages <- c('tidyverse','caret','lubridate','ranger','Hmisc',
                       'scales', 'corrplot','Metrics')
for(p in required_packages){
      if(!require(p,character.only = TRUE)) install.packages(p)
      library(p,character.only = TRUE)
}

### Kaggle dataset from github repo link in submission
df_raw <- read_csv("tmdb-box-office-data.csv")

### split the data into a train and test set
set.seed(57)
trainIndex <- createDataPartition(df_raw$revenue, p = .8, 
                                  list = FALSE, 
                                  times = 1)

train_df <- df_raw[trainIndex,] %>% mutate(label="train")
test_df  <- df_raw[-trainIndex,] %>% mutate(label="test")

all_data <- rbind(train_df, test_df)
dim(all_data)

###############################################################
### Feature cleaning and engineerings
###############################################################

### function to view all features that have NAs
missing_data_fun <- function(df) {
      df_result <- as.data.frame(sapply(df, function(x) sum(is.na(x))))
      df_result <- rownames_to_column(df_result, var = "feature")
      colnames(df_result)[2] <- c("NA_Count")
      df_result %>% dplyr::filter(NA_Count>0) %>% arrange(desc(NA_Count))
}

### if title is missing, replace with original title
all_data <- all_data %>% 
      mutate(title = ifelse(is.na(title), original_title, title))

### replace NA runtime with median runtime
all_data <- all_data %>% 
      mutate(runtime = replace_na(runtime, median(runtime, na.rm=TRUE)))

### convert release date to date type
### replace NA release data with median release date
year_clean <- function(x, year=1917){
      m <- year(x) %% 100
      year(x) <- ifelse(m > year %% 100, 1900+m, 2000+m)
      x
}
all_data <- all_data %>% 
      mutate(release_date_clean = year_clean(as.Date(release_date, format = "%m/%d/%y")),
             release_date_clean = replace_na(release_date_clean, median(release_date_clean, na.rm=TRUE)))

### mode imputation: spoken_languages, status
Mode <- function(x) {
      ux <- unique(x)
      ux[which.max(tabulate(match(x, ux)))]
}
all_data <- all_data %>% 
      mutate(spoken_languages = replace_na(spoken_languages, Mode(spoken_languages)),
             status = replace_na(status, Mode(status)))

### for the remaining features still with NA values replace with Missing:
# belongs_to_collection, homepage, tagline, Keywords, production_companies, 
# production_countries, genres, overview, crew, poster_path
all_data <- all_data %>% 
      mutate_at(missing_data_fun(all_data)$feature, ~replace_na(., "Missing"))

### derive features
all_data <- all_data %>% 
      mutate(pre_process_budget = budget,
             pre_process_budget_available = ifelse(pre_process_budget>0,1,0),
             release_year = year(release_date_clean),
             before_2000_flag = ifelse(year(release_date_clean)<2000,1,0),
             before_1980_flag = ifelse(year(release_date_clean)<1980,1,0),
             release_year_bin = cut2(year(release_date_clean), g=10),
             release_month = month(release_date_clean),
             release_month_day = day(release_date_clean),
             release_week_number = week(release_date_clean),
             release_day_of_week = wday(release_date_clean, label = TRUE),
             release_year_quarter_str = paste0("Quarter","::",quarter(release_date_clean, with_year = FALSE, fiscal_start = 1)),
             title_length = str_length(title),
             belongs_to_collection_flag = ifelse(str_count(belongs_to_collection, "name")>0,1,0),
             tagline_available = ifelse(tagline=="Missing", 0, 1),
             homepage_available = ifelse(homepage=="Missing", 0, 1),
             homepage_disney_flag = ifelse(str_count(homepage, "disney")>0,1,0),
             homepage_sony_flag = ifelse(str_count(homepage, "sony")>0,1,0),
             homepage_warnerbros_flag = ifelse(str_count(homepage, "warnerbros")>0,1,0),
             homepage_focusfeatures_flag = ifelse(str_count(homepage, "focusfeatures")>0,1,0),
             homepage_fox_flag = ifelse(str_count(homepage, "foxmovies")>0 |
                                                str_count(homepage, "foxsearchlight")>0,1,0),
             homepage_magpictures_flag = ifelse(str_count(homepage, "magpictures")>0,1,0),
             homepage_mgm_flag = ifelse(str_count(homepage, ".mgm.")>0,1,0),
             homepage_miramax_flag = ifelse(str_count(homepage, ".miramax.")>0,1,0),
             homepage_facebook_flag = ifelse(str_count(homepage, ".facebook.")>0,1,0),
             genres_count = str_count(genres, "id"),
             production_company_count = str_count(production_companies, "name"),
             production_country_count = str_count(production_countries, "name"),
             spoken_languages_count = str_count(spoken_languages, "name"),
             cast = ifelse(cast=="[]","Missing", cast),
             cast = ifelse(cast=="#N/A","Missing", cast),
             cast_count = str_count(cast, "cast_id"),
             cast_gender_0_count = str_count(cast, "'gender': 0,"),
             cast_gender_1_count = str_count(cast, "'gender': 1,"),
             cast_gender_2_count = str_count(cast, "'gender': 2,"),
             crew = ifelse(crew=="#N/A","Missing", crew),
             crew_count = str_count(crew, "credit_id"),
             director_count = str_count(crew, "job': 'Director', 'name':"),
             producer_count = str_count(crew, "job': 'Producer', 'name':"),
             exec_producer_count = str_count(crew, "'job': 'Executive Producer', 'name':"),
             independent_film_flag = ifelse(str_count(Keywords, "independent film")>0,1,0)
      )

### if zero, replace cast & crew count with median
all_data <- all_data %>% 
      mutate(cast_count = ifelse(cast_count==0,
                                 median(all_data$cast_count), cast_count),
             crew_count = ifelse(crew_count==0, 
                                 median(all_data$crew_count), crew_count))

### build knn model on training data
### use to fill in train and test observations with movie budgets 1000 or less
ctrl <- trainControl(method="repeatedcv",repeats = 3)
knn_budget <- train(budget ~ release_year + cast_count + crew_count + 
                          director_count + exec_producer_count + 
                          production_company_count + production_country_count +
                          independent_film_flag, 
                    data = all_data %>% filter(label=="train" & budget>1000), 
                    method = "knn", 
                    trControl = ctrl, 
                    preProcess = c("center","scale"), 
                    tuneLength = 10)

### use knn model to predict budget for train and test 
### where observations budgets 1000 or less
all_data$budget[all_data$budget<=1000] <- predict(knn_budget, newdata = all_data %>% filter(budget<=1000))

### add log budget feature
all_data <- all_data %>% 
      mutate(log_budget = log(budget))

### get first descriptors listed for what might be important features
all_data <- all_data %>%
      group_by(id) %>%
      mutate(first_genre_listed = replace_na(strsplit(strsplit(genres, "name': '")[[1]][2],"'")[[1]][1],"Missing"),
             first_director_name = replace_na(strsplit(strsplit(crew, "job': 'Director', 'name':")[[1]][2],"'")[[1]][2],"Missing"),
             first_production_company = replace_na(strsplit(strsplit(production_companies, "'name': '")[[1]][2],"'")[[1]][1],"Missing"),
             first_production_country = replace_na(strsplit(strsplit(production_countries, "'name': '")[[1]][2],"'")[[1]][1],"Missing")) %>%
      ungroup()

### use to extract names based on a specified pattern
json_name_parser <- function(pattern, input_string) {
      result <- str_extract_all(input_string, pattern)[[1]]
      result <- str_replace_all(result, pattern, "\\2")
      result <- paste(sort(result), sep="",collapse = ", ")
      result <- if_else(result=="","Missing",result)
      return(result)
}

### use to extract prod company based on a specified pattern
prod_company_parser <- function(pattern, input_string) {
      result <- sort(str_match_all(input_string, prod_companies_pattern)[[1]][,3])
      result <- paste(result, sep="",collapse = ", ")
      result <- if_else(result=="","Missing",result)
      return(result)
}

### regrex patterns
### for some companies entire name is not extracted this could be a future area for improvement
director_pattern <- "('Director', 'name': ')([a-zA-Z]*\\s*[a-zA-Z]*)(')"
collection_pattern <- "(, 'name': ')([a-zA-Z.]*\\s*[a-zA-Z.]*\\s*[a-zA-Z.]*\\s*[a-zA-Z.]*\\s*[a-zA-Z.]*)(')"
prod_companies_pattern <- "(\\{'name': ')([a-zA-Z.]*\\s*[a-zA-Z.]*\\s*[a-zA-Z.]*\\s*[a-zA-Z.]*)"
genres_pattern <- "('name': ')([a-zA-Z]*\\s*[a-zA-Z]*)(')"

all_data <- all_data %>%
      group_by(id) %>%
      mutate(directors_chr = json_name_parser(director_pattern, crew),
             collection_chr = json_name_parser(collection_pattern, belongs_to_collection),
             genres_chr = json_name_parser(genres_pattern, genres),
             production_company_chr = prod_company_parser(prod_companies_pattern, production_companies)) %>%
      ungroup()

top_20_most_popular_first_listed_prod_companies <- all_data %>% 
      filter(label=="train") %>%
      count(first_production_company, sort = TRUE) %>%
      filter(first_production_company!="Missing") %>%
      filter(row_number()<21) %>%
      pull(first_production_company)

popular_first_listed_prod_company <- function(input_str) {
      result <- sum(top_20_most_popular_first_listed_prod_companies
                    %in% unlist(strsplit(input_str, split=", ")))
      return(result)
}

all_data <- all_data %>%
      group_by(id) %>%
      mutate(number_of_popular_first_listed_prod_cos = 
                   popular_first_listed_prod_company(production_company_chr)) %>%
      ungroup()

### reduce levels for first production company
### get the top 200
top_200_most_popular_first_listed_prod_companies <- all_data %>% 
      filter(label=="train") %>%
      count(first_production_company, sort = TRUE) %>%
      filter(first_production_company!="Missing") %>%
      filter(row_number()<201) %>%
      pull(first_production_company)

all_data <- all_data %>%
      group_by(id) %>%
      mutate(first_production_company = ifelse(
            first_production_company %in% top_200_most_popular_first_listed_prod_companies,
            first_production_company, "Other")) %>%
      ungroup()

### NAs know clean from the data
missing_data_fun(all_data)

###############################################################
### Exploratory data analysis
###############################################################

### Top grossing movies in train dataset
all_data %>%
      filter(label=="train") %>%
      arrange(desc(revenue)) %>%
      head(10) %>%
      ggplot(aes(y=reorder(title, revenue), x=revenue/1000000000, fill=revenue/1000000000)) + 
      geom_col() +
      labs(title="Top 10 grossing movies in training dataset",
           y="",
           x="Revenue (Billions)") +
      theme(legend.position = "none")

### Snapshot of top first production companies by movie count
all_data %>%
      filter(label=="train" & first_production_company!="Other") %>%
      group_by(first_production_company) %>%
      summarise(movie_count = n_distinct(imdb_id),
                median_revenue_millions = median(revenue)/1000000) %>%
      top_n(15, movie_count) %>%
      ggplot(aes(x=reorder(first_production_company, movie_count),
                 y=median_revenue_millions,fill=median_revenue_millions)) +
      geom_col() +
      geom_text(aes(label=paste0("n=",movie_count)), vjust=-0.5) +
      theme(legend.position = "none",
            axis.text.x = element_text(angle = 45, hjust = 1),
            plot.margin = margin(1, 1, 1, 2, "cm")) +
      labs(title="Top 15: First Listed Production Companies by Median Movie Revenue",
           x="First Listed Production Company",
           y="Median Revenue (millions)") +
      scale_y_continuous(breaks=seq(0,200,20), expand = expansion(mult = c(0.05, .15)))
      
### Budget zero volume
all_data %>%
      filter(label=="train") %>%
      group_by(budget_1k_or_less_flag = pre_process_budget<=1000) %>%
      summarise(movie_count = n()) %>%
      ungroup() %>%
      mutate(percent_total = movie_count/sum(movie_count)) %>%
      ggplot(aes(y=movie_count, x=budget_1k_or_less_flag, fill=budget_1k_or_less_flag)) +
      geom_col() +
      geom_text(aes(label=percent(percent_total,1)), vjust=-0.5) +
      labs(title="Movie budget less than or equal to 1k in training set.
KNN model used to fill budget value for 28% of training observations.",
           y="Movie Count",
           x="Budget <= 1000?") +
      theme(legend.position = "none") +
      scale_fill_manual(values=c("dodgerblue","grey40"))

### Plot revenue by year and movie count by year
all_data %>%
      filter(label=="train") %>%
      group_by(release_year_bin) %>%
      summarise(movie_count = n_distinct(imdb_id),
                median_budget = median(budget),
                median_revenue = median(revenue),
                avg_budget = mean(budget),
                avg_revenue = mean(revenue)) %>%
      ggplot(aes(x=release_year_bin, group=1)) + 
      geom_line(aes(y=avg_budget), color="grey40") +
      geom_line(aes(y=avg_revenue), color="dodgerblue")

### Plot revenue by year and movie count by year
all_data %>%
      filter(label=="train") %>%
      group_by(release_year_bin) %>%
      summarise(q25 = quantile(budget, .25),
                median = median(budget),
                q75 = quantile(budget, .75)) %>%
      gather(key="metrics", value="value", -release_year_bin) %>%
      ggplot(aes(x=release_year_bin, y=value, group=metrics, color=metrics)) + 
      geom_line() +
      geom_point()

### scatter plot budget vs revenue
all_data %>%
      filter(label=="train") %>%
      ggplot(aes(x=log(budget), y=log(revenue))) +
      geom_point(alpha=0.1)

### plot revenue by year and movie count by year
all_data %>%
      filter(label=="train") %>%
      group_by(release_year_bin) %>%
      summarise(movie_count = n_distinct(imdb_id),
                median_budget = median(budget),
                median_revenue = median(revenue),
                avg_budget = mean(budget),
                avg_revenue = mean(revenue)) %>%
      ggplot(aes(x=release_year_bin, group=1)) + 
      geom_line(aes(y=median_budget), color="grey40") +
      geom_line(aes(y=median_revenue), color="dodgerblue")

### Eda by homepage flag
homepage_df <- all_data %>%
      filter(label=="train") %>%
      select(imdb_id, revenue, contains("homepage_")) %>%
      gather(key = "homepage_var", value="value", -imdb_id, -revenue) %>%
      group_by(homepage_var, value) %>%
      summarise(movie_count = n_distinct(imdb_id),
                movie_row_count = n(),
                median_revenue_millions = median(revenue)/1000000) %>%
      ungroup() %>%
      filter(value==1 | homepage_var=="homepage_available") %>%
      mutate(baseline = ifelse(homepage_var=="homepage_available","Overall", "Homepage Feature Flags"),
             homepage_var = ifelse(homepage_var=="homepage_available" & value==0,
                                   "homepage_NOT_available", homepage_var))

homepage_df %>%
      ggplot(aes(x=reorder(homepage_var, median_revenue_millions),
            y=median_revenue_millions,fill=baseline)) +
      geom_col() +
      geom_text(aes(y=0, label=paste0("n=",movie_count)), vjust=-0.5) +
      facet_grid(.~reorder(baseline, median_revenue_millions), 
                 space = "free_x", scales = "free_x") +
      theme(legend.position = "none",
            axis.text.x = element_text(angle = 45, hjust = 1),
            plot.margin = margin(1, 1, 1, 2, "cm")) +
      scale_y_continuous(breaks=seq(0,400,by=50)) +
      labs(title="Left panel: movies that have a homepage URL tend to have higher box office revenue.
Right panel: well known studios such as disney look to more consistently release high revenue movies.",
           x="",
           y="Median Revenue (millions)")
      
### Seasonal trend of when top movies are released?
all_data %>%
      filter(label=="train") %>%
      group_by(release_month) %>%
      summarise(movie_count = n_distinct(imdb_id),
        median_revenue_millions = median(revenue)/1000000) %>%
      ggplot(aes(x=release_month, y=median_revenue_millions)) +
      geom_col(fill="dodgerblue") +
      geom_text(aes(label=paste0("n=",movie_count)), vjust=-0.5) +
      scale_x_continuous(breaks=1:12) +
      labs(title="Movies released in June, July, and December tend to have higher revenues.
Early summer and holidays are historically when studios look to attract movie goers with films that have tested well.")

### Cor plot for all the numeric features
cor_input <- round(cor(all_data %>% 
                             filter(label=="train") %>%
                             select_if(is.numeric)),2)

### doesn't include fearure names so plot is visible
corrplot(cor_input, method="circle")

###############################################################
### Model Building 
###############################################################

# remove features that aren't target variables or predictors we want to include
# convert character vars to factors
# use log of budget and revenue (which then allows us to use 
# RMSE for optimization)
all_data_features_to_include <- all_data %>%
      mutate(log_revenue = log(revenue)) %>%
      select(-id, -belongs_to_collection, -budget, -genres, 
             -homepage, -imdb_id, -original_title, 
             -popularity, -poster_path, -production_companies,
             -release_date, -production_countries, -spoken_languages,
             -tagline, -title, -Keywords, -cast, -crew,
             -overview, -revenue, - release_date_clean, -first_director_name,
             -directors_chr, -collection_chr, -production_company_chr,
             -pre_process_budget) %>%
      mutate_if(is.character,as.factor)

# training data
training <- all_data_features_to_include %>% 
      filter(label=="train") %>%
      select(-label)

### cross validation
caret_train_control <- trainControl(method = "cv", number = 5)

set.seed(23)
### rpart model
rpart_model <- train(
            log_revenue ~., 
            data = training, 
            method = "rpart",
            metric='RMSE',
            trControl = caret_train_control,
            tuneGrid = data.frame(cp=c(0.0001, .0005, .001, 
                                       0.005, 0.01, 0.025, 0.05, 0.1))
)

### rpart2 model
set.seed(23)
rpart2_model <- train(
            log_revenue ~., 
            data = training, 
            method = "rpart2",
            metric='RMSE',
            trControl = caret_train_control,
            tuneGrid = data.frame(maxdepth=seq(4, 16, by=2))
)

### treebag model
set.seed(23)
treebag_model <- train(
      log_revenue ~., 
      data = training, 
      method = "treebag",
      metric='RMSE',
      trControl = caret_train_control)


rf_grid <- expand.grid(mtry=c(10,20,30),
                       splitrule=c("variance"),
                       min.node.size=c(25, 75, 125))

### Ranger model
set.seed(23)
ranger_model <- train(
            log_revenue ~.,
            data = training,
            method = "ranger",
            metric='RMSE',
            importance = "impurity",
            trControl = caret_train_control,
            tuneGrid = rf_grid)

xgb_grid <- expand.grid(nrounds=100,
                        max_depth=c(2, 4, 6),
                        eta=c(0.2),
                        gamma=c(0,0.1,.5),
                        colsample_bytree=0.75,
                        min_child_weight=0.75,
                        subsample=0.8)

### Xgb model
set.seed(23)
xgb_model <- train(log_revenue ~.,
                      data = training,
                      method = "xgbTree",
                      metric='RMSE',
                      trControl = caret_train_control,
                      tuneGrid=xgb_grid)
xgb_model

### Compare model performances using Caret resample()
models_compare <- resamples(list(rpart1=rpart_model, 
                                 rpart2=rpart2_model,
                                 treebag=treebag_model,
                                 rf=ranger_model,
                                 xgb=xgb_model))

### Summary of the models performances
rmse_cv_results <- data.frame(summary(models_compare)$statistics$RMSE) %>%
      rownames_to_column(var="model") %>%
      rename(Min = Min.,
             Q1 = X1st.Qu.,
             Q3 = X3rd.Qu.,
             Max = Max.)

### Boxplot on target metrics by model
rmse_cv_results %>%
      ggplot(aes(y=model, color=model)) +
      geom_boxplot(aes(xmin=Min,xlower=Q1,xmiddle=Median, 
                       xupper=Q3, xmax=Max), stat = "identity") +
      theme(legend.position = "none") +
      labs(subtitle = "Comparing model RMSE using Caret resample",
           x="RMSE",
           y="Model")

### Show var importance from top model
varImp(xgb_model)$importance %>% 
  as.data.frame() %>%
  rownames_to_column(var = "features") %>%
  top_n(20,Overall) %>%
  rename(importance_score = Overall) %>%
  ggplot(aes(x=reorder(features, importance_score), y=importance_score, fill=importance_score)) +
  geom_col() +
  coord_flip() +
  theme(legend.position = "none") +
  labs(subtitle = "Xgb: Feature Importance Ranking",
       x="Feature",
       y="Feature Importance Score")

###############################################################
### Get pulse check on model performance using holdout test dataset
###############################################################
test <- all_data_features_to_include %>% 
            filter(label=="test") %>%
            select(-label)

### final performance check on test data
caret::RMSE(pred=predict(ranger_model, newdata=test),
            obs= test$log_revenue)

### compare final model vs simple average prediction
caret::RMSE(pred=predict(ranger_model, newdata=test),
            obs= mean(test$log_revenue))






