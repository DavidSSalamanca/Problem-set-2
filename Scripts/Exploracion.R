rm(list = ls())
setwd("C:/Users/USUARIO/Documents/Trabajo/BIG Data/Taller #2")

library(ggplot2)
library(dplyr)
library(gridExtra)
library(stargazer)


hogares_tr <- read.csv("train_hogares_final.csv")
hogares_te <- read.csv("test_hogares_final.csv")

hogares_tr$Promedio_Edad_Ocupados[is.na(hogares_tr$Promedio_Edad_Ocupados)] <- 0
hogares_te$Promedio_Edad_Ocupados[is.na(hogares_te$Promedio_Edad_Ocupados)] <- 0

hogares_tr$P5130[is.na(hogares_tr$P5130)] <- 0
hogares_tr$P5140[is.na(hogares_tr$P5140)] <- 0

hogares_te$P5130[is.na(hogares_te$P5130)] <- 0
hogares_te$P5140[is.na(hogares_te$P5140)] <- 0

# Seleccionar las variables usadas en los modelos en hogares_tr y hogares_te
variables_modelo <- c("Indicador_Hacinamiento_Critico", "Num_Personas_Estudiando", 
                      "Proporción_Estudiando", "Proporción_Ocupados", 
                      "Num_Personas_Trabajando", "Nper", "Vivienda_Arriendo", 
                      "P5010", "Nivel_Educativo_Maximo", "Num_Personas_Media_Superior", 
                      "Num_Mujeres", "Num_Dependientes", "Indicador_Hogar_Jefe_Femenino", 
                      "Promedio_Edad_Ocupados", "P5130", "P5140", 
                      "Vivienda_Posesion_Sin_Titulo", "Promedio_Edad_Ocupados", 
                      "Proporción_Estudiando", "Proporción_Ocupados", 
                      "Num_Personas_Trabajando", "Vivienda_Propia_Pagada")

# Filtrar las variables de interés para hogares_tr y hogares_te
data_filtered_tr <- hogares_tr %>%
  select(all_of(variables_modelo))

data_filtered_te <- hogares_te %>%
  select(all_of(variables_modelo))

# Crear la tabla descriptiva para el conjunto de entrenamiento
stargazer(data_filtered_tr, type = "latex",
          title = "Descriptive Statistics - Train Data",
          summary.stat = c("mean", "sd", "min", "max", "n"),
          digits = 2)

# Crear la tabla descriptiva para el conjunto de prueba
stargazer(data_filtered_te, type = "latex",
          title = "Descriptive Statistics - Test Data",
          summary.stat = c("mean", "sd", "min", "max", "n"),
          digits = 2)



# Gráfico 2: Relación entre número de dependientes y hacinamiento crítico (nuevo ajuste de escala Y)
ggplot(hogares_tr, aes(x = Num_Dependientes, fill = factor(Indicador_Hacinamiento_Critico))) +
  geom_bar(position = "dodge") +
  scale_x_continuous(breaks = seq(0, max(hogares_tr$Num_Dependientes), by = 1)) +  # Mantener los valores enteros en X
  coord_cartesian(ylim = c(0, 10000)) +  # Ajuste del eje Y para mantener el gráfico claro
  labs(title = "Número de Dependientes vs. Hacinamiento Crítico", 
       x = "Número de Dependientes", y = "Frecuencia", fill = "Hacinamiento Crítico") +
  theme_minimal()


# Filtrar el nivel educativo para excluir el valor 9
hogares_tr_filtered <- subset(hogares_tr, Nivel_Educativo_Maximo != 9)

# Gráfico 1: Relación entre el Nivel Educativo Máximo y el Número de Dependientes (sin el nivel 9)
ggplot(hogares_tr_filtered, aes(x = factor(Nivel_Educativo_Maximo), y = Num_Dependientes)) +
  geom_boxplot(fill = "steelblue") +
  labs(title = "Relación entre Nivel Educativo Máximo y Número de Dependientes", 
       x = "Nivel Educativo Máximo", y = "Número de Dependientes") +
  theme_minimal()

# Gráfico 2: Proporción de Ocupados por Vivienda (Propia Pagada vs. Alquiler)
ggplot(hogares_tr, aes(x = factor(Vivienda_Propia_Pagada), y = Proporción_Ocupados)) +
  geom_boxplot(fill = "darkseagreen") +
  labs(title = "Proporción de Ocupados por Tipo de Vivienda", 
       x = "Vivienda Propia Pagada (1 = Sí, 0 = No)", y = "Proporción de Ocupados") +
  theme_minimal()

# Gráfico 3: Relación entre Proporción de Ocupados e Indicador de Hacinamiento Crítico
ggplot(hogares_tr, aes(x = factor(Indicador_Hacinamiento_Critico), y = Proporción_Ocupados)) +
  geom_boxplot(fill = "gold") +
  labs(title = "Proporción de Ocupados vs. Hacinamiento Crítico", 
       x = "Hacinamiento Crítico (1 = Sí, 0 = No)", y = "Proporción de Ocupados") +
  theme_minimal()
#grafico 4
ggplot(hogares_tr, aes(x = factor(Indicador_Hogar_Jefe_Femenino), y = Proporción_Ocupados, fill = factor(Indicador_Hogar_Jefe_Femenino))) +
  geom_boxplot() +
  labs(title = "Proporción de Ocupados por Sexo del Jefe de Hogar", 
       x = "Jefe de Hogar Femenino", y = "Proporción de Ocupados", fill = "Sexo") +
  theme_minimal()


# Reemplazar el nivel educativo 9 por 1 antes de graficar
hogares_tr$Nivel_Educativo_Maximo <- ifelse(hogares_tr$Nivel_Educativo_Maximo == 9, 1, hogares_tr$Nivel_Educativo_Maximo)
#grafico 5
# Crear el gráfico con los cambios aplicados
ggplot(hogares_tr, aes(x = factor(Nivel_Educativo_Maximo), fill = factor(Indicador_Hacinamiento_Critico))) +
  geom_bar(position = "fill") +
  labs(title = "Distribución del Hacinamiento Crítico por Nivel Educativo Máximo", 
       x = "Nivel Educativo Máximo", y = "Proporción", fill = "Hacinamiento Crítico") +
  theme_minimal()




