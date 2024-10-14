rm(list = ls())
setwd("C:/Users/USUARIO/Documents/Trabajo/BIG Data/Taller #2")

install.packages("caret")
install.packages("ROSE")
install.packages("Metrics")
library(dplyr)
library(readr)
library(caret)
library(Metrics) 

hogares_tr <- read.csv("train_hogares_final.csv")
hogares_te <- read.csv("test_hogares_final.csv")

# Reemplazar los NA en la variable Promedio_Edad_Ocupados por 0
hogares_tr$Promedio_Edad_Ocupados[is.na(hogares_tr$Promedio_Edad_Ocupados)] <- 0
hogares_te$Promedio_Edad_Ocupados[is.na(hogares_te$Promedio_Edad_Ocupados)] <- 0

hogares_tr$Promedio_Edad_Ocupados[is.na(hogares_tr$Promedio_Edad_Ocupados)] <- 0
hogares_te$Promedio_Edad_Ocupados[is.na(hogares_te$Promedio_Edad_Ocupados)] <- 0

train_data$p5130[is.na(train_data$p5130)] <- 0
train_data$p5140[is.na(train_data$p5140)] <- 0

test_data$p5130[is.na(test_data$p5130)] <- 0
test_data$p5140[is.na(test_data$p5140)] <- 0


modelo_probit_simple <- glm(Pobre ~ Nper + Indicador_Hacinamiento_Critico, 
                            family = binomial(link = "probit"), data = train_data)

# Ver el resumen del modelo
summary(modelo_probit_simple)


# Modelo Probit 1: Vivienda y Ocupación
modelo_probit_1 <- glm(Pobre ~ Vivienda_Arriendo + Vivienda_Propia_Pagada + Indicador_Hacinamiento_Critico + Proporción_Ocupados, 
                       family = binomial(link = "probit"), data = hogares_tr)
summary(modelo_probit_1)


# Modelo Probit 2: Educación y Dependientes
modelo_probit_2 <- glm(Pobre ~ Num_Dependientes + Proporción_Estudiando + Max_Basica_Secundaria + Max_Media + Max_Superior, 
                       family = binomial(link = "probit"), data = hogares_tr)
summary(modelo_probit_2)

# Modelo Probit 3: Demografía y Género
modelo_probit_3 <- glm(Pobre ~ Proporción_Mujeres + Indicador_Hogar_Jefe_Femenino + Promedio_Edad_Ocupados + Num_Personas_Estudiando, 
                       family = binomial(link = "probit"), data = hogares_tr)
summary(modelo_probit_3)

modelo_probit_4 <- glm(Pobre ~ Vivienda_Posesion_Sin_Titulo + 
                              Indicador_Hacinamiento_Critico + 
                              Num_Personas_Media_Superior, 
                            family = binomial(link = "probit"), data = hogares_tr)

# Resumen del modelo
summary(modelo_probit_4)

# Modelo 5: Probit con todas las dummies de tipo de vivienda y educación máxima (MAL MODELO)
#modelo_5 <- glm(Pobre ~ Vivienda_Propia_Pagada + Vivienda_Pagando + Vivienda_Arriendo + Vivienda_Usufructo + 
#                 Vivienda_Posesion_Sin_Titulo + Vivienda_Otra +
#                  Max_Ninguno + Max_Preescolar + Max_Basica_Primaria + Max_Basica_Secundaria + 
 #                 Max_Media + Max_Superior, 
  #              family = binomial(link = "probit"), 
   #             data = hogares_tr)

#summary(modelo_5)


# Crear el modelo 6
modelo_6 <- glm(Pobre ~ Vivienda_Propia_Pagada + 
                  Indicador_Hacinamiento_Critico + 
                  Num_Personas_Trabajando + 
                  Num_Personas_Estudiando + 
                  Num_Personas_Trabajando:Num_Personas_Estudiando, 
                data = hogares_tr, 
                family = binomial)

# Resumen del modelo
summary(modelo_6)

# Ajustar el modelo probit con las variables adicionales
modelo_7 <- glm(Pobre ~ Vivienda_Propia_Pagada + 
                       Indicador_Hacinamiento_Critico + 
                       Num_Personas_Trabajando + 
                       Num_Personas_Estudiando + 
                       Num_Personas_Trabajando:Num_Personas_Estudiando + 
                       Num_Dependientes + 
                       Num_Dependientes:Num_Personas_Trabajando, 
                     family = binomial(link = "probit"), 
                     data = hogares_tr)

# Resumen del modelo
summary(modelo_7)


set.seed(10101)  # Para reproducibilidad

# 1. Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
train_index <- createDataPartition(hogares_tr$Pobre, p = 0.7, list = FALSE)
train_data <- hogares_tr[train_index, ]
test_data <- hogares_tr[-train_index, ]

# Verificar el desbalanceo en la clase Pobre
table(train_data$Pobre)

# Modelo Probit 1: Vivienda y Ocupación
modelo_probit_1 <- glm(Pobre ~ Vivienda_Arriendo + Vivienda_Propia_Pagada + Indicador_Hacinamiento_Critico + Proporción_Ocupados, 
                       family = binomial(link = "probit"), data = train_data)

# Modelo Probit 2: Educación y Dependientes
modelo_probit_2 <- glm(Pobre ~ Num_Dependientes + Proporción_Estudiando + Max_Basica_Secundaria + Max_Media + Max_Superior, 
                       family = binomial(link = "probit"), data = train_data)

# Modelo Probit 3: Demografía y Género
modelo_probit_3 <- glm(Pobre ~ Proporción_Mujeres + Indicador_Hogar_Jefe_Femenino + Promedio_Edad_Ocupados + Num_Personas_Estudiando, 
                       family = binomial(link = "probit"), data = train_data)

# Modelo Probit 4: Vivienda y Educación
modelo_probit_4 <- glm(Pobre ~ Vivienda_Posesion_Sin_Titulo + 
                         Indicador_Hacinamiento_Critico + 
                         Num_Personas_Media_Superior, 
                       family = binomial(link = "probit"), data = train_data)

# Modelo Probit 6: Vivienda, Hacinamiento y Trabajo + Interacción
modelo_probit_6 <- glm(Pobre ~ Vivienda_Propia_Pagada + 
                         Indicador_Hacinamiento_Critico + 
                         Num_Personas_Trabajando + 
                         Num_Personas_Estudiando + 
                         Num_Personas_Trabajando:Num_Personas_Estudiando, 
                       family = binomial(link = "probit"), data = train_data)

# Modelo Probit 7: Con interacción entre dependientes y ocupación
modelo_probit_7 <- glm(Pobre ~ Vivienda_Propia_Pagada + 
                         Indicador_Hacinamiento_Critico + 
                         Num_Personas_Trabajando + 
                         Num_Personas_Estudiando + 
                         Num_Personas_Trabajando:Num_Personas_Estudiando + 
                         Num_Dependientes + 
                         Num_Dependientes:Num_Personas_Trabajando, 
                       family = binomial(link = "probit"), data = train_data)

# Modelo Probit 8: Interacción entre dependientes y hacinamiento
modelo_probit_8 <- glm(Pobre ~ Vivienda_Propia_Pagada + 
                         Indicador_Hacinamiento_Critico + 
                         Num_Personas_Trabajando + 
                         Num_Personas_Estudiando + 
                         Num_Personas_Trabajando:Num_Personas_Estudiando + 
                         Num_Dependientes + 
                         Num_Dependientes:Indicador_Hacinamiento_Critico, 
                       family = binomial(link = "probit"), data = train_data)

# Modelo Probit 9: Con proporción de mujeres
modelo_probit_9 <- glm(Pobre ~ Vivienda_Propia_Pagada + 
                         Indicador_Hacinamiento_Critico + 
                         Num_Personas_Trabajando + 
                         Num_Personas_Estudiando + 
                         Num_Personas_Trabajando:Num_Personas_Estudiando + 
                         Num_Dependientes + 
                         Proporción_Mujeres, 
                       family = binomial(link = "probit"), data = train_data)


#(el mejor) modelo_probit_10 <- glm(Pobre ~ Nper + Indicador_Hacinamiento_Critico + 
          #               Proporción_Ocupados * Num_Dependientes + 
           #              Num_Personas_Media_Superior * Max_Superior, 
            #            family = binomial(link = "probit"), data = train_data)

# Modelo Probit 10 con la interacción entre Num_Personas_Media_Superior y Max_Superior
modelo_probit_10 <- glm(Pobre ~ Nper + Indicador_Hacinamiento_Critico + 
                          Proporción_Ocupados * Num_Dependientes + 
                          Num_Personas_Media_Superior + Max_Superior +
                          Num_Personas_Media_Superior * Max_Superior, 
                        family = binomial(link = "probit"), data = train_data)



# Función para calcular RMSE, Accuracy y F1-Score
calcular_metricas <- function(model, test_data) {
  # Predicciones de probabilidad
  prob_pred <- predict(model, newdata = test_data, type = "response")
  
  # Convertir las probabilidades en clases (umbral 0.5)
  class_pred <- ifelse(prob_pred > 0.5, 1, 0)
  
  # Calcular RMSE
  rmse_value <- sqrt(mean((class_pred - test_data$Pobre)^2))
  
  # Calcular Accuracy
  accuracy_value <- mean(class_pred == test_data$Pobre)
  
  # Calcular el F1-Score
  conf_matrix <- confusionMatrix(factor(class_pred), factor(test_data$Pobre))
  f1_value <- conf_matrix$byClass["F1"]
  
  return(list(rmse = rmse_value, accuracy = accuracy_value, f1 = f1_value))
}

# Calcular métricas para cada modelo
resultado_probit_1 <- calcular_metricas(modelo_probit_1, test_data)
resultado_probit_2 <- calcular_metricas(modelo_probit_2, test_data)
resultado_probit_3 <- calcular_metricas(modelo_probit_3, test_data)
resultado_probit_4 <- calcular_metricas(modelo_probit_4, test_data)
resultado_probit_6 <- calcular_metricas(modelo_probit_6, test_data)
resultado_probit_7 <- calcular_metricas(modelo_probit_7, test_data)
resultado_probit_8 <- calcular_metricas(modelo_probit_8, test_data)
resultado_probit_9 <- calcular_metricas(modelo_probit_9, test_data)
resultado_probit_10 <- calcular_metricas(modelo_probit_10, test_data)

cat("Modelo Probit 1 - RMSE:", resultado_probit_1$rmse, "- Accuracy:", resultado_probit_1$accuracy, "- F1:", resultado_probit_1$f1, "\n")
cat("Modelo Probit 2 - RMSE:", resultado_probit_2$rmse, "- Accuracy:", resultado_probit_2$accuracy, "- F1:", resultado_probit_2$f1, "\n")
cat("Modelo Probit 3 - RMSE:", resultado_probit_3$rmse, "- Accuracy:", resultado_probit_3$accuracy, "- F1:", resultado_probit_3$f1, "\n")
cat("Modelo Probit 4 - RMSE:", resultado_probit_4$rmse, "- Accuracy:", resultado_probit_4$accuracy, "- F1:", resultado_probit_4$f1, "\n")
cat("Modelo Probit 6 - RMSE:", resultado_probit_6$rmse, "- Accuracy:", resultado_probit_6$accuracy, "- F1:", resultado_probit_6$f1, "\n")
cat("Modelo Probit 7 - RMSE:", resultado_probit_7$rmse, "- Accuracy:", resultado_probit_7$accuracy, "- F1:", resultado_probit_7$f1, "\n")
cat("Modelo Probit 8 - RMSE:", resultado_probit_8$rmse, "- Accuracy:", resultado_probit_8$accuracy, "- F1:", resultado_probit_8$f1, "\n")
cat("Modelo Probit 9 - RMSE:", resultado_probit_9$rmse, "- Accuracy:", resultado_probit_9$accuracy, "- F1:", resultado_probit_9$f1, "\n")
cat("Modelo Probit 10 - RMSE:", resultado_probit_10$rmse, "- Accuracy:", resultado_probit_10$accuracy, "- F1:", resultado_probit_10$f1, "\n")


# Hacer predicciones con el Modelo Probit 2 en la base de test (hogares_te)
predicciones_probit_2 <- predict(modelo_probit_2, newdata = hogares_te, type = "response")

# Convertir las predicciones en clases binarias (0 o 1) utilizando un umbral de 0.5
prediccion_clase_2 <- ifelse(predicciones_probit_2 > 0.5, 1, 0)

# Crear un dataframe con el ID y las predicciones del Modelo Probit 2
resultado_probit_2 <- data.frame(id = hogares_te$id, pobre = prediccion_clase_2)

# Guardar las predicciones en un CSV con un nombre reflejando el método y hiperparámetros
write.csv(resultado_probit_2, "prediccion_probit2_Num_Dependientes_Educacion.csv", row.names = FALSE, quote = FALSE)

# Hacer predicciones con el Modelo Probit 7 en la base de test (hogares_te)
predicciones_probit_7 <- predict(modelo_probit_7, newdata = hogares_te, type = "response")

# Convertir las predicciones en clases binarias (0 o 1) utilizando un umbral de 0.5
prediccion_clase_7 <- ifelse(predicciones_probit_7 > 0.5, 1, 0)

# Crear un dataframe con el ID y las predicciones del Modelo Probit 7
resultado_probit_7 <- data.frame(id = hogares_te$id, pobre = prediccion_clase_7)

# Guardar las predicciones en un CSV con un nombre reflejando el método y hiperparámetros
write.csv(resultado_probit_7, "prediccion_probit7_Hacinamiento_Educacion_Interaccion.csv", row.names = FALSE, quote = FALSE)

# Hacer predicciones con el Modelo Probit 10 en la base de test (hogares_te)
predicciones_probit_10 <- predict(modelo_probit_10, newdata = hogares_te, type = "response")

# Convertir las predicciones en clases binarias (0 o 1) utilizando un umbral de 0.5
prediccion_clase_10 <- ifelse(predicciones_probit_10 > 0.5, 1, 0)

# Crear un dataframe con el ID y las predicciones del Modelo Probit 10
resultado_probit_10 <- data.frame(id = hogares_te$id, pobre = prediccion_clase_10)

# Guardar las predicciones en un CSV con un nombre reflejando el método y las interacciones añadidas
write.csv(resultado_probit_10, "prediccion_probit10_Nper_Hacinamiento_Educacion_Interacciones.csv", row.names = FALSE, quote = FALSE)




pred_probit2 <- read.csv("prediccion_probit2_Num_Dependientes_Educacion.csv")
pred_probit7 <- read.csv("prediccion_probit7_Hacinamiento_Educacion_Interaccion.csv")



