# Pacotes -----------------------------------------------------------------

library(tidymodels)
library(forecast)
library(caret)
library(lightgbm)
library(bonsai)
library(tibble)
library(ggplot2)
library(timetk)
library(tidyr)
library(lubridate)

setwd("D:/Unb/TCC")

# Dados -------------------------------------------------------------------

datas <- seq(as.Date("2011-02-01"), as.Date("2024-09-01"), by = "month")

load(file="covariates_Train_ts.RData")
dados <- data.frame(covariates_Train_ts)
load(file="covariates_Test_ts.RData")
load(file="dependent_variables_ts.RData")
load(file="responses_series_level.RData")

dependentVariables <- data.frame(dependent_variables_ts)


inpc <- dependentVariables$y_adjusted.inpc_br
inpc_TS <- ts(inpc, start = c(2011, 2), end = c(2024,9), frequency = 12)

#inpc_diferenca <- c(dependent_variables_ts[,1])
#inpc_diferencaTS <- ts(inpc_diferenca, start = c(2011, 1), end = c(2024,9), frequency = 12)

#inpc_var_perc_TS <- (inpc_diferencaTS / inpc_TS) * 100
#inpc_var_perc_TS <- na.omit(inpc_var_perc_TS)
#inpc_var_perc <- c(inpc_var_perc_TS)

serieINPC <- read.csv("inpc_2011_a_2025.csv", sep = ";", 
                      header = TRUE,
                      dec = ",", 
                      encoding = "UTF-8")

serieINPC <- serieINPC %>%
  mutate(Data = ymd(paste0(Data, "-01")))

ggplot(serieINPC, aes(x = Data, y = Variacao.Percentual.Mensal)) +
  geom_line(color = "darkblue", linewidth = 1) +
  labs(
    x = "Ano",
    y = "Variação Percentual (%)"
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  scale_y_continuous(breaks = seq(-0.5, 2.0, by = 0.5)) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggplot(serieINPC, aes(x = Data, y = Indice.INPC)) +
  geom_line(color = "darkblue", linewidth = 1) +
  labs(
    x = "Ano",
    y = "Índice INPC"
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  scale_y_continuous(breaks = seq(3500, 8000, by = 1000)) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

dataset <- data.frame(datas,inpc,dados)

splits <- initial_split(dataset, prop = 0.75)
train <- training(splits)
test <- testing(splits)

# Verificar NA's no train e test
sum(is.na(train))
sum(is.na(test))

# Análise exploratória INPC -----------------------------------------------

plot(inpc_TS, 
     xlab = "Ano", 
     ylab = "Valor do INPC", 
     col = "steelblue", 
     lwd = 2)
grid()  # Adiciona grade para melhorar a visualização

# Adicionar linha horizontal no 0
abline(h = 0, col = "black", lwd = 2, lty = 2)  # Linha preta tracejada

df_inpc <- data.frame(
  Data = as.Date(time(inpc_var_perc_2022)),
  Variacao = as.numeric(inpc_var_perc_2022)
)

# Regularização Lasso -----------------------------------------------------

y <- inpc_TS
xreg <- dados

lassoFit <-  glmnet::cv.glmnet(x=model.matrix(~.-1,data=xreg), y=as.vector(y), family="gaussian", intercept = FALSE,
                               alpha =1) 

lfc <- as.vector(coef(lassoFit, s = 'lambda.min')[,1]!= 0)[-1]
namesCoef <- colnames(xreg)[lfc]
print(namesCoef)

covariaveisLasso <-  as.data.frame(xreg[,which(colnames(xreg)%in% c(namesCoef))])


# LightGBM ---------------------------------------------------------------------
set.seed(202046640)

datasetLasso <- data.frame(inpc,covariaveisLasso)

splits <- initial_split(datasetLasso, prop = 0.75)
train <- training(splits)
test <- testing(splits)

# Engine
lightGBM_modelo <- boost_tree(trees = tune(), tree_depth = tune(), min_n = tune(), learn_rate = tune()) %>% 
  set_engine("lightgbm") %>% 
  set_mode("regression")

# Recipe
lightGBM_recipe <- recipe(inpc ~ ., data = train) %>% 
  step_normalize(all_predictors())

# Workflow
lightGBM_wflow <- workflow() %>% 
  add_recipe(lightGBM_recipe) %>% 
  add_model(lightGBM_modelo)

# Cross Validation
val_set <- vfold_cv(train, v = 5)

# Training
lightGBM_train <- lightGBM_wflow %>% 
  tune_grid(resamples = val_set,
            grid = 100,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(yardstick::rmse, yardstick::rsq)
)

# Mostrar os melhores modelos
lightGBM_train %>% show_best(n = 10)

# validacao cruzada light gbm h -------------------------------------------
set.seed(202046640)

# Criar um dataframe para armazenar os resultados do RMSE e MAE
metrics_results <- data.frame(Horizonte = integer(), RMSE = numeric(), MAE = numeric())

completo <- rbind(train,test)

# Loop sobre diferentes horizontes de previsão
for (h in 1:39) {
  
  # Criar validação cruzada com rolling origin para horizonte h
  val_set_h <- rolling_origin(
    data = completo,
    initial = floor(nrow(completo) * 0.7),
    assess = h,  
    cumulative = TRUE 
  )
  
  # Criar listas para armazenar RMSE e MAE de cada fold
  rmse_folds <- c()
  mae_folds <- c()
  
  for (split in val_set_h$splits) {
    train_split <- training(split)  
    test_split <- testing(split)    
    
    lightGBM_model <- boost_tree(
      trees = 127, 
      min_n = 8, 
      tree_depth = 12, 
      learn_rate = 0.0219
    ) %>% 
      set_engine("lightgbm") %>% 
      set_mode("regression")
    
    lightGBM_wflow <- workflow() %>%
      add_recipe(lightGBM_recipe) %>% 
      add_model(lightGBM_model)
    
    lightGBM_fit <- fit(lightGBM_wflow, data = train_split)
    
    preds <- predict(lightGBM_fit, new_data = test_split) %>%
      bind_cols(test_split)
    
    # Calcular RMSE e MAE
    rmse_fold <- rmse(preds, truth = inpc, estimate = .pred) %>% pull(.estimate)
    mae_fold <- mae(preds, truth = inpc, estimate = .pred) %>% pull(.estimate)
    
    rmse_folds <- c(rmse_folds, rmse_fold)
    mae_folds <- c(mae_folds, mae_fold)
  }
  
  # Calcular RMSE e MAE médios dos folds para o horizonte h
  mean_rmse_h <- mean(rmse_folds)
  mean_mae_h <- mean(mae_folds)
  
  metrics_results <- rbind(metrics_results, data.frame(Horizonte = h, RMSE = mean_rmse_h, MAE = mean_mae_h))
}

print(metrics_results)
rmseGBMLasso <- metrics_results[, c("Horizonte", "RMSE")]

# Aplicação Séries Temporais --------------------------------------------
library(xgboost)
library(tidymodels)
library(modeltime)
library(lubridate)
library(timetk)

# Transformando em série temporal

# EDA do INPC
summary(inpc_TS)
plot(decompose(inpc_TS))

plot(decompose(inpc_TS, type = 'mult'))

# ACF e PACF
acf(inpc_TS, lag.max = 60)

pacf(inpc_TS)

# Teste de hipótese de estacionariedade
adf.test(inpc_diferencaTS)

# ACF e PACF
acf(inpc_diferencaTS)
pacf(inpc_diferencaTS)

# Auto Arima
Arima_INPC <- auto.arima(inpc_TS,
                         stepwise = FALSE,
                         approximation = FALSE)


ArimaForecast <- forecast(Arima_INPC, h = 12)
plot(ArimaForecast)

# Parâmetros do SARIMA por auto.arima ------------------------------------------------------------------
xreg_corrigido <- as.matrix(covariaveisLasso)

modelo_auto <- auto.arima(inpc_TS, xreg = xreg_corrigido,seasonal = TRUE, stepwise = FALSE, approximation = FALSE, trace = TRUE)

melhores_modelos <- data.frame(
  Modelo = c(
    "ARIMA(4,1,0)(0,0,1)[12]",
    "ARIMA(4,1,1)(0,0,1)[12]",
    "ARIMA(4,1,2)(0,0,1)[12]",
    "ARIMA(4,1,0)(0,0,2)[12]",
    "ARIMA(4,1,1)(0,0,2)[12]",
    "ARIMA(4,1,2)(0,0,2)[12]",
    "ARIMA(1,1,0)(0,0,1)[12]",
    "ARIMA(1,1,1)(0,0,1)[12]",
    "ARIMA(2,1,1)(0,0,1)[12]",
    "ARIMA(1,1,2)(0,0,1)[12]"
  ),AIC = c(
    1431.996,
    1432.990,
    1433.750,
    1433.827,
    1434.751,
    1435.675,
    1475.698,
    1476.360,
    1476.538,
    1476.694))

library(forecast)

covariaveisLasso <- ts(covariaveisLasso, start = c(2011, 2), end = c(2024,9), frequency = 12)


# 4 melhores sarimax ------------------------------------------------------

# Definir a função de previsão para tsCV
forecast_sarimax1 <- function(y, h, xreg, newxreg) {
  fit <- Arima(y, order = c(4, 1, 0), seasonal = list(order = c(0, 0, 1), period = 12)
               ,xreg = as.matrix(xreg))

  forecast(fit, h = h, xreg = as.matrix(newxreg))
}

# Executar validação cruzada para h = 1 até 6
erros_predicao <- tsCV(inpc_TS, forecast_sarimax1, h = 43, xreg = covariaveisLasso)

# Calcular RMSE médio
rmseSarimaxLasso1 <- sqrt(colMeans(erros_predicao^2, na.rm = TRUE))

print(rmseSarimaxLasso)

# Definir a função de previsão para tsCV
forecast_sarimax2 <- function(y, h, xreg, newxreg) {
  # Ajustar modelo SARIMAX
  fit <- Arima(y, order = c(4, 1, 1), seasonal = list(order = c(0, 0, 1), period = 12)
               ,xreg = as.matrix(xreg))
  
  forecast(fit, h = h, xreg = as.matrix(newxreg))
}

# Executar validação cruzada para h = 1 até 6
erros_predicao <- tsCV(inpc_TS, forecast_sarimax2, h = 43, xreg = covariaveisLasso)

# Calcular RMSE médio
rmseSarimaxLasso2 <- sqrt(colMeans(erros_predicao^2, na.rm = TRUE))

# Definir a função de previsão para tsCV
forecast_sarimax3 <- function(y, h, xreg, newxreg) {
  # Ajustar modelo SARIMAX
  fit <- Arima(y, order = c(4, 1, 2), seasonal = list(order = c(0, 0, 1), period = 12)
               ,xreg = as.matrix(xreg))
  
  forecast(fit, h = h, xreg = as.matrix(newxreg))
}

# Executar validação cruzada para h = 1 até 6
erros_predicao <- tsCV(inpc_TS, forecast_sarimax3, h = 39, xreg = covariaveisLasso, initial = 114)

# Calcular RMSE médio
rmseSarimaxLasso3 <- sqrt(colMeans(erros_predicao^2, na.rm = TRUE))
rmseSarimaxLasso <- data.frame(Horizonte = 1:39, RMSE = rmseSarimaxLasso3)

# Definir a função de previsão para tsCV
forecast_sarimax4 <- function(y, h, xreg, newxreg) {
  # Ajustar modelo SARIMAX
  fit <- Arima(y, order = c(4, 1, 0), seasonal = list(order = c(0, 0, 2), period = 12)
               ,xreg = as.matrix(xreg))
  
  forecast(fit, h = h, xreg = as.matrix(newxreg))
}

# Executar validação cruzada para h = 1 até 6
erros_predicao <- tsCV(inpc_TS, forecast_sarimax4, h = 43, xreg = covariaveisLasso)

# Calcular RMSE médio
rmseSarimaxLasso4 <- sqrt(colMeans(erros_predicao^2, na.rm = TRUE))
rmseSarimaxLasso4 <- data.frame(Horizonte = 1:43, RMSE = rmseSarimaxLasso4)

print(rmseSarimaxLasso4)

# Definir a função de previsão para tsCV
forecast_sarimax5 <- function(y, h, xreg, newxreg) {
  # Ajustar modelo SARIMAX(4,1,1)(0,0,2,12)
  fit <- Arima(y, order = c(4, 1, 1), seasonal = list(order = c(0, 0, 2), period = 12)
               ,xreg = as.matrix(xreg))
  
  forecast(fit, h = h, xreg = as.matrix(newxreg))
}

# Executar validação cruzada para h = 1 até 6
erros_predicao <- tsCV(inpc_TS, forecast_sarimax5, h = 43, xreg = covariaveisLasso)

# Calcular RMSE médio
rmseSarimaxLasso5 <- sqrt(colMeans(erros_predicao^2, na.rm = TRUE))

# XGBoost -----------------------------------------------------------------
set.seed(202046640)

XGBoost_modelo <- boost_tree(trees = tune(), tree_depth = tune(),
                              loss_reduction = tune(), min_n = tune(),
                              learn_rate = tune(), sample_size = tune(),
                              stop_iter = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

# Recipe
XGBoost_recipe <- recipe(inpc ~ ., data = train) %>% 
  step_normalize(all_predictors())

# Workflow
XGBoost_wflow <- workflow() %>% 
  add_recipe(XGBoost_recipe) %>% 
  add_model(XGBoost_modelo)

# Cross Validation
val_set <- vfold_cv(train, v = 5)

# Training
XGBoost_train <- XGBoost_wflow %>% 
  tune_grid(resamples = val_set,
            grid = 100,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(yardstick::rmse, yardstick::rsq)
  )

# Mostrar os melhores modelos
XGBoost_train %>% show_best(n = 10)

metrics_results_xgboost <- data.frame(Horizonte = integer(), RMSE = numeric(), MAE = numeric())

completo <- rbind(train,test) 

# Loop sobre diferentes horizontes de previsão
for (h in 1:39) {
  
  # Criar validação cruzada com rolling origin para horizonte h
  val_set_h <- rolling_origin(
    data = completo,
    initial = floor(nrow(completo) * 0.7),
    assess = h,  
    cumulative = TRUE 
  )
  
  rmse_folds <- c()
  mae_folds <- c()
  
  for (split in val_set_h$splits) {
    train_split <- training(split)  
    test_split <- testing(split)   
    
    # Criar modelo LightGBM ajustado ao conjunto de treino atual
    XGBoost_model <- boost_tree(
      trees = 1496, 
      min_n = 13, 
      tree_depth = 10, 
      learn_rate = 0.00518,
      loss_reduction = 2.21,
      sample_size = 0.960,
      stop_iter = 5
    ) %>% 
      set_engine("xgboost") %>% 
      set_mode("regression")
    
    # Criar receita adaptada ao split atual
    XGBoost_recipe_split <- recipe(inpc ~ ., data = train_split) %>% 
      step_normalize(all_predictors())
    
    # Criar workflow para o fold atual
    XGBoost_wflow <- workflow() %>%
      add_recipe(XGBoost_recipe_split) %>% 
      add_model(XGBoost_model)
    
    # Treinar o modelo apenas com os dados disponíveis no fold
    XGBoost_fit <- fit(XGBoost_wflow, data = train_split)
    
    # Fazer previsões no conjunto de teste do fold
    preds <- predict(XGBoost_fit, new_data = test_split) %>%
      bind_cols(test_split)
    
    # Calcular RMSE e MAE
    rmse_fold <- rmse(preds, truth = inpc, estimate = .pred) %>% pull(.estimate)
    mae_fold <- mae(preds, truth = inpc, estimate = .pred) %>% pull(.estimate)
    
    rmse_folds <- c(rmse_folds, rmse_fold)
    mae_folds <- c(mae_folds, mae_fold)
  }
  
  # Calcular RMSE e MAE médios dos folds para o horizonte h
  mean_rmse_h <- mean(rmse_folds)
  mean_mae_h <- mean(mae_folds)

  metrics_results_xgboost <- rbind(metrics_results_xgboost, data.frame(Horizonte = h, RMSE = mean_rmse_h, MAE = mean_mae_h))
}

rmseXGBoostLasso <- metrics_results_xgboost[, c("Horizonte", "RMSE")]

# Modelos ETS -------------------------------------------------------------

#transformando em ts
responseTS <- ts(responses_series_level$y_adjusted.inpc_br, 
                 start = c(2011, 2), 
                 frequency = 12)
decomposicaoTS <- decompose(responseTS)
plot(decomposicaoTS)

forecast_ets_fixo <- function(x, h) {
  fit_ets <- ets(x, model = "ANN")
  return(forecast(fit_ets, h = h))
}

erros_ETS_fixo <- tsCV(inpc_TS, forecast_ets_fixo, h = 39)
rmseETS_Fixo <- sqrt(colMeans(erros_ETS_fixo^2, na.rm = TRUE))
rmseETS_Fixo <- data.frame(Horizonte = 1:39, RMSE = rmseETS_Fixo)

# Comparação RMSE dos modelos --------------------------------------------------
df_rmse <- data.frame(
  horizonte = 1:39,
  rmseETS_Fixo$RMSE,
  rmseXGBoostLasso$RMSE,
  rmseGBMLasso$RMSE,
  rmseSarimaxLasso$RMSE
)
colnames(df_rmse)[2] <- "ETS"
colnames(df_rmse)[3] <- "XGBoost"
colnames(df_rmse)[4] <- "GBM"
colnames(df_rmse)[5] <- "Sarimax"
df_long <- pivot_longer(df_rmse,
                        cols = -horizonte,
                        names_to = "Modelo",
                        values_to = "RMSE")

# Gráfico de comparação do RMSE por Horizonte
ggplot(df_long, aes(x = horizonte, y = RMSE, color = Modelo, group = Modelo)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = c("ETS" = "yellow",
                                "XGBoost" = "red",
                                "GBM" = "blue",
                                "Sarimax" = "green")) +
  scale_y_continuous(limits = c(0, max(df_long$RMSE) * 1.1), 
                     breaks = seq(0, max(df_long$RMSE)+5, by = 5)) +
  labs(title = "Comparação do RMSE por Horizonte de Previsão",
       x = "Horizonte de Previsão (1 a 39)",
       y = "Raiz do Erro Quadrático Médio (RMSE)",
       color = "Modelo") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.title = element_text(face = "bold"),
        legend.title = element_text(face = "bold"))

# Previsão para o futuro GBM --------------------------------------------------
lightGBM_model <- boost_tree(
  trees = 127, 
  min_n = 8, 
  tree_depth = 12, 
  learn_rate = 0.0219
) %>%
  set_engine("lightgbm") %>% 
  set_mode("regression")

lightGBM_wflow <- workflow() %>%
  add_recipe(lightGBM_recipe) %>% 
  add_model(lightGBM_model)

lightGBM_fit <- fit(lightGBM_wflow, data = completo)

covariaveisTeste <- as.data.frame(covariates_Test_ts[,which(colnames(covariates_Test_ts)%in% c(namesCoef))])

predsGBM <- predict(lightGBM_fit, new_data = covariaveisTeste)
Predito <- data.frame(predsGBM$.pred)

# Previsão para o futuro XGBoost --------------------------------------------------
XGBoost_model <- boost_tree(
  trees = 936, 
  min_n = 5, 
  tree_depth = 1, 
  learn_rate = 0.0336,
  loss_reduction = 0.226,
  sample_size = 0.829,
  stop_iter = 12
) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

XGBoost_wflow <- workflow() %>%
  add_recipe(XGBoost_recipe) %>% 
  add_model(XGBoost_model)

XGBoost_fit <- fit(XGBoost_wflow, data = completo)

predsXGBoost <- predict(XGBoost_fit, new_data = covariaveisTeste)

Predito$XGBoost <- c(predsXGBoost$.pred)

# Previsão para o futuro SARIMAX --------------------------------------------------
fit <- Arima(y, order = c(4, 1, 2), seasonal = list(order = c(0, 0, 1), period = 12)
               ,xreg = as.matrix(covariaveisLasso))

covariaveisTesteSARIMA <-  as.data.frame(covariaveisTeste[,which(colnames(covariaveisTeste)%in% c(namesCoef))])

predsSARIMA <- forecast(fit, xreg = as.matrix(covariaveisTeste))

Predito$SARIMAX <- c(predsSARIMA$mean)

# Previsão para o futuro ETS --------------------------------------------------
fit_ets <- forecast::ets(inpc_TS, model = "AAN")
previsao_ets_39 <- forecast::forecast(fit_ets, h = 39)

print(previsao_ets_39)
plot(previsao_ets_39)
Predito$ETS <- c(previsao_ets_39$mean)

Mensal <- data.frame(
  ETS = c(responses_series_level$y_adjusted.inpc_br[165],Predito$ETS),
  GBM = c(responses_series_level$y_adjusted.inpc_br[165],Predito$predsGBM..pred),
  XGBOOST = c(responses_series_level$y_adjusted.inpc_br[165],Predito$XGBoost),
  SARIMAX = c(responses_series_level$y_adjusted.inpc_br[165],Predito$SARIMAX)
)

# Cálculo do índice acumulado
Indice <- data.frame(
  GBM = Predito$predsGBM..pred,
  XGBOOST = Predito$XGBoost,
  SARIMAX = Predito$SARIMAX
)

Indice <- data.frame(
  ETS = c(responses_series_level$y_adjusted.inpc_br[165],Predito$ETS),
  GBM = c(responses_series_level$y_adjusted.inpc_br[165],Predito$predsGBM..pred),
  XGBOOST = c(responses_series_level$y_adjusted.inpc_br[165],Predito$XGBoost),
  SARIMAX = c(responses_series_level$y_adjusted.inpc_br[165],Predito$SARIMA)
)
Indice$ETS <- cumsum(Indice$ETS)
Indice$GBM <- cumsum(Indice$GBM)
Indice$XGBOOST <- cumsum(Indice$XGBOOST)
Indice$SARIMAX <- cumsum(Indice$SARIMA)


# TabelaRMSEs -------------------------------------------------------------
 ConsolidadoRMSE <- data.frame(
   horizonte = 1:39,
   round(rmseETS_Fixo$RMSE,2),
   round(rmseXGBoostLasso$RMSE,2),
   round(rmseGBMLasso$RMSE,2),
   round(rmseSarimaxLasso$RMSE,2)
 )

write.csv(ConsolidadoRMSE, file = "ConsolidadoRMSE.csv", row.names = FALSE) 

Out24Mai25 <- read.csv("D:/Unb/TCC/Out24Mai25.csv", sep = ";")
Out24Mai25

PrevXReal <- cbind(Out24Mai25[,2],Indice[1:9,])
colnames(PrevXReal)[1] <- "Real"

write.csv(PrevXReal, file = "PrevXReal.csv", row.names = FALSE)

PrevXReal_percentual <- PrevXReal %>%
  mutate(
    across(
      .cols = everything(),
      .fns = ~ case_when(
        row_number() == 1 ~ 0.48,
        is.na(lag(.)) ~ NA_real_,
        lag(.) == 0 ~ ifelse(. == 0, 0, NA_real_),
        TRUE ~ ((. - lag(.)) / lag(.)) * 100
      ),
      .names = "{.col}_variacaoPct" # Adiciona sufixo ao nome da coluna original
    )
  )

PrevXReal_percentual <- PrevXReal_percentual[,6:10]
PrevXReal_percentual <- PrevXReal_percentual %>%
  mutate(across(everything(), ~ round(., 2)))
PrevXReal_percentual <- PrevXReal_percentual[, c("Real_variacaoPct","ETS_variacaoPct", "XGBOOST_variacaoPct", "GBM_variacaoPct", "SARIMA_variacaoPct")]

write.csv(PrevXReal_percentual, file = "PrevXReal_percentual.csv")

# Comparação das previsões com os valores reais
rotulos_eixo_x <- c("Out/24", "Nov/24", "Dez/24", "Jan/25", "Fev/25", "Mar/25", "Abr/25", "Mai/25")

ggplot(PrevXReal, aes(x = 1:nrow(PrevXReal))) +
  geom_line(aes(y = ETS, color = "ETS"), linewidth = 1) +
  geom_line(aes(y = Real, color = "Real"), linewidth = 1) + 
  geom_line(aes(y = GBM, color = "GBM"), linewidth = 1) +
  geom_line(aes(y = XGBOOST, color = "XGBOOST"), linewidth = 1) +
  geom_line(aes(y = SARIMAX, color = "SARIMAX"), linewidth = 1) +
  geom_point(aes(y = ETS, color = "ETS"), size = 2.5) +
  geom_point(aes(y = Real, color = "Real"), size = 2.5) + 
  geom_point(aes(y = GBM, color = "GBM"), size = 2.5) + 
  geom_point(aes(y = XGBOOST, color = "XGBOOST"), size = 2.5) + 
  geom_point(aes(y = SARIMAX, color = "SARIMAX"), size = 2.5) + 
  labs(title = "Comparação das Previsões com os Valores Reais",
       x = "Período",
       y = "Índice INPC",
       color = "Legenda") +
  scale_color_manual(values = c("ETS" = "yellow","Real" = "black", "GBM" = "blue", "XGBOOST" = "red", "SARIMAX" = "green")) +
  scale_x_continuous(
    breaks = 1:nrow(PrevXReal), 
    labels = rotulos_eixo_x     
  ) +
  scale_y_continuous(
    limits = c(7150, 7500),
    breaks = seq(7150, 7500, by = 50)
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top",
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.title = element_text(face = "bold"),
        legend.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

modelos <- c("ETS_variacaoPct", "XGBOOST_variacaoPct", "GBM_variacaoPct", "SARIMAX_variacaoPct")

# Calcular MAE de cada modelo 
mae_resultados <- sapply(modelos, function(modelo) {
  mean(abs(PrevXReal_percentual[[modelo]] - PrevXReal_percentual$Real_variacaoPct), na.rm = TRUE)
})

mae_resultados
write.csv(mae_resultados, file = "PrevXReal_MAE.csv")

# Acumulado INPC Out24 a Mai25 --------------------------------------------

Acumulado <- PrevXReal_percentual[4:9,1:5]
Acumulado$Real_variacaoPct <- Acumulado$Real_variacaoPct/100
Acumulado$ETS_variacaoPct <- Acumulado$ETS_variacaoPct/100
Acumulado$XGBOOST_variacaoPct <- Acumulado$XGBOOST_variacaoPct/100
Acumulado$GBM_variacaoPct <- Acumulado$GBM_variacaoPct/100
Acumulado$SARIMAX_variacaoPct <- Acumulado$SARIMAX_variacaoPct/100

AcumuladoReal <- (prod(1 + Acumulado$Real_variacaoPct) - 1) * 100
AcumuladoETS <- (prod(1+ Acumulado$ETS_variacaoPct)-1)*100
AcumuladoXGBoost <- (prod(1+ Acumulado$XGBOOST_variacaoPct)-1)*100
AcumuladoGBM <- (prod(1+ Acumulado$GBM_variacaoPct)-1)*100
AcumuladoSARIMAX <- (prod(1+ Acumulado$SARIMAX_variacaoPct)-1)*100
AcumuladoPercentual <- data.frame(
  INPCModelos = c("INPC","ETS", "XGBoost", "GBM", "SARIMAX"),
  Acumulado = c(AcumuladoReal,AcumuladoETS, AcumuladoXGBoost, AcumuladoGBM, AcumuladoSARIMAX)
)
