# Cargar librerías necesarias
library(glmnet)  # Para modelos Elastic Net
library(caret)   # Para partición y evaluación
library(Metrics) # Para calcular RMSE
library(dplyr)   # Para manipulación de datos

hogares_tr <- read.csv("train_hogares_final.csv")
hogares_te <- read.csv("test_hogares_final.csv")

hogares_tr$Promedio_Edad_Ocupados[is.na(hogares_tr$Promedio_Edad_Ocupados)] <- 0
hogares_te$Promedio_Edad_Ocupados[is.na(hogares_te$Promedio_Edad_Ocupados)] <- 0

hogares_tr$P5130[is.na(hogares_tr$P5130)] <- 0
hogares_tr$P5140[is.na(hogares_tr$P5140)] <- 0

hogares_te$P5130[is.na(hogares_te$P5130)] <- 0
hogares_te$P5140[is.na(hogares_te$P5140)] <- 0

set.seed(10101)  # Semilla para reproducibilidad
train_index <- createDataPartition(hogares_tr$Pobre, p = 0.7, list = FALSE)
train_data <- hogares_tr[train_index, ]
test_data <- hogares_tr[-train_index, ]

# Preparar las variables predictoras y la variable dependiente
X_train <- model.matrix(Pobre ~ Nper + Indicador_Hacinamiento_Critico + 
                          Proporción_Ocupados * Num_Dependientes + 
                          Num_Personas_Media_Superior + Max_Superior + 
                          Num_Personas_Media_Superior * Max_Superior, 
                        data = train_data)[,-1]  # Eliminamos la primera columna de intercepto

y_train <- train_data$Pobre

X_test <- model.matrix(Pobre ~ Nper + Indicador_Hacinamiento_Critico + 
                         Proporción_Ocupados * Num_Dependientes + 
                         Num_Personas_Media_Superior + Max_Superior + 
                         Num_Personas_Media_Superior * Max_Superior, 
                       data = test_data)[,-1]

y_test <- test_data$Pobre

# Establecer la validación cruzada para encontrar el mejor lambda
set.seed(10101)
cv_model <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", nfolds = 10)

# Mejor valor de lambda encontrado por validación cruzada
best_lambda <- cv_model$lambda.min
cat("Mejor lambda:", best_lambda, "\n")

# Ajustar el modelo final con el mejor lambda
modelo_elastic_net <- glmnet(X_train, y_train, alpha = 0.5, family = "binomial", lambda = best_lambda)

# Predecir en el conjunto de prueba
prob_pred <- predict(modelo_elastic_net, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral de 0.5)
pred_class <- ifelse(prob_pred > 0.5, 1, 0)

# Calcular Accuracy y RMSE
accuracy_value <- mean(pred_class == y_test)
rmse_value <- sqrt(mean((pred_class - y_test)^2))

# Calcular el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(y_test))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")


# Modificar el modelo Elastic Net con nuevas interacciones
modelo_elastic_net_modificado <- glmnet(
  X_train, 
  y_train, 
  alpha = 0.5,  # Elastic Net (mezcla de Ridge y Lasso)
  lambda = best_lambda
)

# Interacciones nuevas
X_train_interact <- model.matrix(Pobre ~ Nper + Indicador_Hacinamiento_Critico +
                                   Proporción_Ocupados * Indicador_Hacinamiento_Critico +
                                   Num_Personas_Media_Superior * Proporción_Ocupados +
                                   Num_Dependientes * Indicador_Hacinamiento_Critico +
                                   Num_Mujeres * Proporción_Ocupados +
                                   Max_Superior * Proporción_Ocupados, 
                                 data = train_data)[,-1]  # Eliminar la columna de intercepto

X_test_interact <- model.matrix(Pobre ~ Nper + Indicador_Hacinamiento_Critico +
                                  Proporción_Ocupados * Indicador_Hacinamiento_Critico +
                                  Num_Personas_Media_Superior * Proporción_Ocupados +
                                  Num_Dependientes * Indicador_Hacinamiento_Critico +
                                  Num_Mujeres * Proporción_Ocupados +
                                  Max_Superior * Proporción_Ocupados, 
                                data = test_data)[,-1]

# Ajustar el modelo Elastic Net modificado
modelo_elastic_net_modificado <- cv.glmnet(
  X_train_interact, 
  y_train, 
  alpha = 0.35,  # Elastic Net
  family = "binomial"
)

# Seleccionar el mejor lambda basado en la validación cruzada
best_lambda <- modelo_elastic_net_modificado$lambda.min

# Predicciones en los datos de test
pred_prob_elastic_net_modificado <- predict(modelo_elastic_net_modificado, s = best_lambda, newx = X_test_interact, type = "response")
pred_class_elastic_net_modificado <- ifelse(pred_prob_elastic_net_modificado > 0.5, 1, 0)

# Evaluación del modelo
rmse_value_modificado <- sqrt(mean((pred_class_elastic_net_modificado - y_test)^2))
accuracy_value_modificado <- mean(pred_class_elastic_net_modificado == y_test)

# Calcular el F1-Score
conf_matrix_modificado <- confusionMatrix(factor(pred_class_elastic_net_modificado), factor(y_test))
f1_value_modificado <- conf_matrix_modificado$byClass["F1"]

# Mostrar resultados
cat("Elastic Net Modificado - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net Modificado:", rmse_value_modificado, "\n")
cat("Accuracy del modelo Elastic Net Modificado:", accuracy_value_modificado, "\n")
cat("F1-Score del modelo Elastic Net Modificado:", f1_value_modificado, "\n")



# Crear nuevas interacciones para el modelo Elastic Net
X_train_interact_2 <- model.matrix(Pobre ~ Nper + Indicador_Hacinamiento_Critico +
                                     Proporción_Ocupados * Max_Superior +
                                     Num_Dependientes * Num_Personas_Media_Superior +
                                     Indicador_Hacinamiento_Critico * Max_Media +
                                     Num_Mujeres + 
                                     Proporción_Ocupados, 
                                   data = train_data)[,-1]

X_test_interact_2 <- model.matrix(Pobre ~ Nper + Indicador_Hacinamiento_Critico +
                                    Proporción_Ocupados * Max_Superior +
                                    Num_Dependientes * Num_Personas_Media_Superior +
                                    Indicador_Hacinamiento_Critico * Max_Media +
                                    Num_Mujeres + 
                                    Proporción_Ocupados, 
                                  data = test_data)[,-1]

# Diferentes valores de alpha para Elastic Net
alpha_values <- seq(0, 1, by = 0.1)  # Probar alpha desde 0 (Ridge) hasta 1 (Lasso)
best_alpha <- 0
best_lambda <- 0
best_rmse <- Inf  # Inicializar el mejor RMSE con un valor alto
best_accuracy <- 0
best_f1 <- 0

# Iterar sobre los valores de alpha
for (alpha in alpha_values) {
  # Ajustar el modelo Elastic Net para cada valor de alpha
  modelo_elastic_net_alpha <- cv.glmnet(
    X_train_interact_2, 
    y_train, 
    alpha = alpha,  # Cambiar el valor de alpha en cada iteración
    family = "binomial"
  )
  
  # Obtener el mejor lambda para el valor actual de alpha
  lambda_alpha <- modelo_elastic_net_alpha$lambda.min
  
  # Predicciones en el conjunto de test
  pred_prob_alpha <- predict(modelo_elastic_net_alpha, s = lambda_alpha, newx = X_test_interact_2, type = "response")
  pred_class_alpha <- ifelse(pred_prob_alpha > 0.5, 1, 0)
  
  # Calcular RMSE
  rmse_alpha <- sqrt(mean((pred_class_alpha - y_test)^2))
  
  # Calcular Accuracy
  accuracy_alpha <- mean(pred_class_alpha == y_test)
  
  # Calcular F1-Score
  conf_matrix_alpha <- confusionMatrix(factor(pred_class_alpha), factor(y_test))
  f1_alpha <- conf_matrix_alpha$byClass["F1"]
  
  # Guardar el mejor modelo
  if (rmse_alpha < best_rmse) {
    best_rmse <- rmse_alpha
    best_accuracy <- accuracy_alpha
    best_f1 <- f1_alpha
    best_alpha <- alpha
    best_lambda <- lambda_alpha
  }
}

# Resultados del mejor modelo Elastic Net
cat("Elastic Net - Mejor alpha:", best_alpha, "\n")
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del mejor modelo Elastic Net:", best_rmse, "\n")
cat("Accuracy del mejor modelo Elastic Net:", best_accuracy, "\n")
cat("F1-Score del mejor modelo Elastic Net:", best_f1, "\n")

## SEGUNDO MEJOR##
# Preparar la matriz de predictores con las interacciones propuestas, incluyendo las nuevas variables
X <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                    Num_Personas_Estudiando * Proporción_Estudiando + 
                    Proporción_Ocupados * Num_Personas_Trabajando + 
                    Proporción_Ocupados + Nper + 
                    Vivienda_Arriendo * P5010 +  # Incluir la interacción con P5010
                    Nivel_Educativo_Maximo + 
                    Num_Personas_Media_Superior, 
                  data = train_data)[,-1]  # Eliminar el intercepto

y <- train_data$Pobre  # Variable dependiente

# Ajustar el modelo Elastic Net con validación cruzada
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X, y, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X, y, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * P5010 +  # Incluir la interacción con P5010
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior, 
                       data = test_data)[,-1]  # Eliminar el intercepto

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

## PRIMERA PREDICCION###

# Preparar la matriz de predictores para los datos de prueba (hogares_te)
X_test <- model.matrix(~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * P5010 +  # Incluir la interacción con P5010
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior, 
                       data = hogares_te)[,-1]  # Eliminar el intercepto

# Realizar predicciones en los datos de prueba (hogares_te)
predicciones_prob_te <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class_te <- ifelse(predicciones_prob_te > 0.5, 1, 0)

# Crear un dataframe con el ID y las predicciones
resultado_final <- data.frame(id = hogares_te$id, pobre = pred_class_te)
# Cambiar el nombre de la columna predicha de "s0" a "pobre"
colnames(resultado_final)[2] <- "pobre"
# Guardar las predicciones en un CSV en el formato adecuado
write.table(resultado_final, "prediccion_elastic_net_hacinamiento_estudiantes.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)


##tercer MEJOR PERO NO POR MUCHO#
# Preparar la matriz de predictores para los datos de entrenamiento con las nuevas interacciones y variables
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) +  # Incluir la interacción con el cuadrado de P5010
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando, 
                        data = train_data)[,-1]  # Eliminar el intercepto

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba, incluyendo las nuevas interacciones y variables
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) +  # Incluir la interacción con el cuadrado de P5010
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando, 
                       data = test_data)[,-1]  # Eliminar el intercepto

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

##El mejor##
# Preparar la matriz de predictores para los datos de entrenamiento con las nuevas interacciones y variables
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) +  # Interacción con el cuadrado de P5010
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando +
                          Num_Mujeres +  # Añadir el número de mujeres
                          Num_Dependientes * Indicador_Hogar_Jefe_Femenino,  # Interacción dependientes x jefe femenino
                        data = train_data)[,-1]  # Eliminar el intercepto

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba, incluyendo las nuevas interacciones y variables
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) +  # Interacción con el cuadrado de P5010
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando +
                         Num_Mujeres +  # Añadir el número de mujeres
                         Num_Dependientes * Indicador_Hogar_Jefe_Femenino,  # Interacción dependientes x jefe femenino
                       data = test_data)[,-1]  # Eliminar el intercepto

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

## EL mejor v.2##
# Preparar la matriz de predictores para los datos de entrenamiento con las nuevas interacciones y variables
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) +  # Interacción con el cuadrado de P5010
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando +
                          Num_Mujeres +  # Añadir el número de mujeres
                          Num_Dependientes * Indicador_Hogar_Jefe_Femenino +  # Interacción dependientes x jefe femenino
                          I(Promedio_Edad_Ocupados^2),  # Añadir la edad al cuadrado
                        data = train_data)[,-1]  # Eliminar el intercepto

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba, incluyendo las nuevas interacciones y variables
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) +  # Interacción con el cuadrado de P5010
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando +
                         Num_Mujeres +  # Añadir el número de mujeres
                         Num_Dependientes * Indicador_Hogar_Jefe_Femenino +  # Interacción dependientes x jefe femenino
                         I(Promedio_Edad_Ocupados^2),  # Añadir la edad al cuadrado
                       data = test_data)[,-1]  # Eliminar el intercepto

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

################################################### el Mejor V3###
# Separar los datos de entrenamiento en entrenamiento y prueba (70/30)
set.seed(10101)  # Semilla para reproducibilidad
train_index <- createDataPartition(hogares_tr$Pobre, p = 0.7, list = FALSE)
train_data <- hogares_tr[train_index, ]
test_data <- hogares_tr[-train_index, ]

# Preparar la matriz de predictores con las interacciones propuestas
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) + 
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando +
                          Num_Mujeres + 
                          Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                          I(Promedio_Edad_Ocupados^2) + 
                          P5130 + P5140, 
                        data = train_data)[,-1]

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) + 
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando +
                         Num_Mujeres + 
                         Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                         I(Promedio_Edad_Ocupados^2) + 
                         P5130 + P5140, 
                       data = test_data)[,-1]

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

############################################ EL MEJOR V4   #########################
# Preparar la matriz de predictores con las interacciones propuestas
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Estudiando * P5130 +  # Interacción con P5130
                          Proporción_Estudiando * P5140 +  # Interacción con P5140
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) + 
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando +
                          Num_Mujeres + 
                          Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                          I(Promedio_Edad_Ocupados^2) + 
                          P5130 + P5140, 
                        data = train_data)[,-1]

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.8, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.8, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Estudiando * P5130 +  # Interacción con P5130
                         Proporción_Estudiando * P5140 +  # Interacción con P5140
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) + 
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando +
                         Num_Mujeres + 
                         Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                         I(Promedio_Edad_Ocupados^2) + 
                         P5130 + P5140, 
                       data = test_data)[,-1]

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")


############################################ EL MEJOR V5 #########
# Preparar la matriz de predictores con las interacciones propuestas
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) + 
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando +
                          Num_Mujeres + 
                          Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                          I(Promedio_Edad_Ocupados^2) + 
                          P5130 + P5140 + 
                          P5130 * Vivienda_Posesion_Sin_Titulo,  # Añadir la interacción de P5130 y Vivienda Posesión Sin Título
                        data = train_data)[,-1]

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) + 
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando +
                         Num_Mujeres + 
                         Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                         I(Promedio_Edad_Ocupados^2) + 
                         P5130 + P5140 + 
                         P5130 * Vivienda_Posesion_Sin_Titulo,  # Añadir la interacción de P5130 y Vivienda Posesión Sin Título
                       data = test_data)[,-1]

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

############################SEGUNDA PREDSICCION ########################

# Preparar la matriz de predictores para los datos de hogares_te (sin la variable Pobre)
X_te <- model.matrix(~ Indicador_Hacinamiento_Critico + 
                       Num_Personas_Estudiando * Proporción_Estudiando + 
                       Proporción_Ocupados * Num_Personas_Trabajando + 
                       Proporción_Ocupados + Nper + 
                       Vivienda_Arriendo * I(P5010^2) + 
                       Nivel_Educativo_Maximo + 
                       Num_Personas_Media_Superior * Num_Personas_Trabajando +
                       Num_Mujeres + 
                       Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                       I(Promedio_Edad_Ocupados^2) + 
                       P5130 + P5140 + 
                       P5130 * Vivienda_Posesion_Sin_Titulo,  # Añadir la interacción de P5130 y Vivienda Posesión Sin Título
                     data = hogares_te)[,-1]  # Eliminar el intercepto

# Realizar predicciones en los datos de hogares_te
predicciones_prob_te <- predict(elastic_net_model, newx = X_te, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class_te <- ifelse(predicciones_prob_te > 0.5, 1, 0)

# Crear un dataframe con el ID y las predicciones
resultado_final_te <- data.frame(id = hogares_te$id, pobre = pred_class_te)

# Cambiar el nombre de la columna predicha de "s0" a "pobre"
colnames(resultado_final_te)[2] <- "pobre"

# Guardar las predicciones en un CSV en el formato adecuado
write.table(resultado_final_te, "prediccion_elastic_net_hacinamiento_estudiantes.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)


#################################### EL MEJOR V6 ################
# Preparar la matriz de predictores con las interacciones propuestas
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) + 
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando +
                          Num_Mujeres + 
                          Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                          Num_Dependientes * Nivel_Educativo_Maximo +  # Añadir la interacción nueva
                          I(Promedio_Edad_Ocupados^2) + 
                          P5130 + P5140 + 
                          P5130 * Vivienda_Posesion_Sin_Titulo,  # Añadir la interacción de P5130 y Vivienda Posesión Sin Título
                        data = train_data)[,-1]

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) + 
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando +
                         Num_Mujeres + 
                         Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                         Num_Dependientes * Nivel_Educativo_Maximo +  # Añadir la interacción nueva
                         I(Promedio_Edad_Ocupados^2) + 
                         P5130 + P5140 + 
                         P5130 * Vivienda_Posesion_Sin_Titulo,  # Añadir la interacción de P5130 y Vivienda Posesión Sin Título
                       data = test_data)[,-1]

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

################################### MEJOR V7 ##########################
# Preparar la matriz de predictores con las interacciones propuestas
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) + 
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando +
                          Num_Mujeres + 
                          Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                          Num_Dependientes * Nivel_Educativo_Maximo +  # Interacción entre dependientes y nivel educativo
                          Proporción_Mujeres * Num_Personas_Media_Superior +  # Añadir la nueva interacción
                          I(Promedio_Edad_Ocupados^2) + 
                          P5130 + P5140 + 
                          P5130 * Vivienda_Posesion_Sin_Titulo,  # Añadir la interacción de P5130 y Vivienda Posesión Sin Título
                        data = train_data)[,-1]

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) + 
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando +
                         Num_Mujeres + 
                         Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                         Num_Dependientes * Nivel_Educativo_Maximo +  # Interacción entre dependientes y nivel educativo
                         Proporción_Mujeres * Num_Personas_Media_Superior +  # Nueva interacción
                         I(Promedio_Edad_Ocupados^2) + 
                         P5130 + P5140 + 
                         P5130 * Vivienda_Posesion_Sin_Titulo,  # Añadir la interacción de P5130 y Vivienda Posesión Sin Título
                       data = test_data)[,-1]

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

# Preparar la matriz de predictores con las interacciones propuestas, incluyendo Vivienda_Propia_Pagada
X_train <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                          Num_Personas_Estudiando * Proporción_Estudiando + 
                          Proporción_Ocupados * Num_Personas_Trabajando + 
                          Proporción_Ocupados + Nper + 
                          Vivienda_Arriendo * I(P5010^2) + 
                          Vivienda_Propia_Pagada +  # Añadir la variable Vivienda_Propia_Pagada
                          Nivel_Educativo_Maximo + 
                          Num_Personas_Media_Superior * Num_Personas_Trabajando +
                          Num_Mujeres + 
                          Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                          Num_Dependientes * Nivel_Educativo_Maximo +  # Interacción entre dependientes y nivel educativo
                          Proporción_Mujeres * Num_Personas_Media_Superior +  # Interacción añadida
                          I(Promedio_Edad_Ocupados^2) + 
                          P5130 + P5140 + 
                          P5130 * Vivienda_Posesion_Sin_Titulo,  # Interacción añadida
                        data = train_data)[,-1]

# Variable dependiente para los datos de entrenamiento
y_train <- train_data$Pobre

# Ajustar el modelo Elastic Net con validación cruzada para optimizar lambda
set.seed(10101)  # Semilla para reproducibilidad
cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.5, family = "binomial", type.measure = "class", nfolds = 10)

# Obtener el mejor lambda
best_lambda <- cv_fit$lambda.min

# Ajustar el modelo final con el mejor lambda
elastic_net_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda, family = "binomial")

# Preparar la matriz de predictores para los datos de prueba
X_test <- model.matrix(Pobre ~ Indicador_Hacinamiento_Critico + 
                         Num_Personas_Estudiando * Proporción_Estudiando + 
                         Proporción_Ocupados * Num_Personas_Trabajando + 
                         Proporción_Ocupados + Nper + 
                         Vivienda_Arriendo * I(P5010^2) + 
                         Vivienda_Propia_Pagada +  # Añadir la variable Vivienda_Propia_Pagada
                         Nivel_Educativo_Maximo + 
                         Num_Personas_Media_Superior * Num_Personas_Trabajando +
                         Num_Mujeres + 
                         Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                         Num_Dependientes * Nivel_Educativo_Maximo +  # Interacción entre dependientes y nivel educativo
                         Proporción_Mujeres * Num_Personas_Media_Superior +  # Interacción añadida
                         I(Promedio_Edad_Ocupados^2) + 
                         P5130 + P5140 + 
                         P5130 * Vivienda_Posesion_Sin_Titulo,  # Interacción añadida
                       data = test_data)[,-1]

# Realizar predicciones en los datos de prueba
predicciones_prob <- predict(elastic_net_model, newx = X_test, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class <- ifelse(predicciones_prob > 0.5, 1, 0)

# Calcular el RMSE
rmse_value <- sqrt(mean((pred_class - test_data$Pobre)^2))

# Calcular el Accuracy
accuracy_value <- mean(pred_class == test_data$Pobre)

# Calcular la matriz de confusión y el F1-Score
conf_matrix <- confusionMatrix(factor(pred_class), factor(test_data$Pobre))
f1_value <- conf_matrix$byClass["F1"]

# Mostrar los resultados
cat("Elastic Net - Mejor lambda:", best_lambda, "\n")
cat("RMSE del modelo Elastic Net:", rmse_value, "\n")
cat("Accuracy del modelo Elastic Net:", accuracy_value, "\n")
cat("F1-Score del modelo Elastic Net:", f1_value, "\n")

################################ tercera prediccion ################

# Preparar la matriz de predictores para los datos de hogares_te (sin la variable Pobre)
X_te <- model.matrix(~ Indicador_Hacinamiento_Critico + 
                       Num_Personas_Estudiando * Proporción_Estudiando + 
                       Proporción_Ocupados * Num_Personas_Trabajando + 
                       Proporción_Ocupados + Nper + 
                       Vivienda_Arriendo * I(P5010^2) + 
                       Vivienda_Propia_Pagada +  # Añadir la variable Vivienda_Propia_Pagada
                       Nivel_Educativo_Maximo + 
                       Num_Personas_Media_Superior * Num_Personas_Trabajando +
                       Num_Mujeres + 
                       Num_Dependientes * Indicador_Hogar_Jefe_Femenino + 
                       Num_Dependientes * Nivel_Educativo_Maximo +  # Interacción entre dependientes y nivel educativo
                       Proporción_Mujeres * Num_Personas_Media_Superior +  # Interacción añadida
                       I(Promedio_Edad_Ocupados^2) + 
                       P5130 + P5140 + 
                       P5130 * Vivienda_Posesion_Sin_Titulo,  # Añadir la interacción de P5130 y Vivienda Posesión Sin Título
                     data = hogares_te)[,-1]  # Eliminar el intercepto

# Realizar predicciones en los datos de hogares_te
predicciones_prob_te <- predict(elastic_net_model, newx = X_te, type = "response")

# Convertir las probabilidades en clases (umbral 0.5)
pred_class_te <- ifelse(predicciones_prob_te > 0.5, 1, 0)

# Crear un dataframe con el ID y las predicciones
resultado_final_te <- data.frame(id = hogares_te$id, pobre = pred_class_te)

# Cambiar el nombre de la columna predicha de "s0" a "pobre"
colnames(resultado_final_te)[2] <- "pobre"
# Guardar las predicciones en un CSV en el formato adecuado
write.table(resultado_final_te, "prediccion_elastic_net_final_v5.csv", sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)
