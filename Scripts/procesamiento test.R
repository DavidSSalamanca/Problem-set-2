# Cargar las bases de datos nuevamente
rm(list = ls())
setwd("C:/Users/USUARIO/Documents/Trabajo/BIG Data/Taller #2")

library(dplyr)
library(readr)

# Cargar las bases de datos
train_personas <- read.csv("train_personas.csv")
train_hogares <- read.csv("train_hogares.csv")
test_personas <- read.csv("test_personas.csv")
test_hogares <- read.csv("test_hogares.csv")

# Asegurar que las variables 'id' estén en el mismo formato
train_personas$id <- as.factor(train_personas$id)
train_hogares$id <- as.factor(train_hogares$id)
test_personas$id <- as.factor(test_personas$id)
test_hogares$id <- as.factor(test_hogares$id)

# Reemplazar valores NA en las variables de test y train para el nivel educativo y ocupación
train_personas$P6210[is.na(train_personas$P6210)] <- 1
test_personas$P6210[is.na(test_personas$P6210)] <- 1

# Asignar valor 6 ("Otra actividad") a los valores NA de la variable P6240
train_personas$P6240[is.na(train_personas$P6240)] <- 6
test_personas$P6240[is.na(test_personas$P6240)] <- 6

# Asignar valor 6 ("Otra") a los valores NA en la variable P5090 para tipo de vivienda
train_hogares$P5090[is.na(train_hogares$P5090)] <- 6
test_hogares$P5090[is.na(test_hogares$P5090)] <- 6

# Función para agregar todas las variables a nivel de hogar
agregar_variables_hogar <- function(df_personas, df_hogares) {
  
  # Número de personas trabajando en el hogar (P6240 == 1)
  num_personas_trabajando <- df_personas %>%
    group_by(id) %>%
    summarise(Num_Personas_Trabajando = sum(P6240 == 1, na.rm = TRUE))
  
  # Proporción de personas ocupadas en el hogar (P6240 == 1 indica que está trabajando)
  ocupados_hogar <- df_personas %>%
    group_by(id) %>%
    summarise(Proporción_Ocupados = mean(P6240 == 1, na.rm = TRUE)) 
  
  # Promedio de edad de las personas ocupadas en el hogar
  promedio_edad_ocupados <- df_personas %>%
    filter(P6240 == 1) %>%
    group_by(id) %>%
    summarise(Promedio_Edad_Ocupados = mean(P6040, na.rm = TRUE))
  
  # Dummies basadas en el tipo de vivienda (P5090) y un indicador si el hogar está en arriendo o no
  df_hogares <- df_hogares %>%
    mutate(
      Vivienda_Propia_Pagada = ifelse(P5090 == 1, 1, 0),
      Vivienda_Pagando = ifelse(P5090 == 2, 1, 0),
      Vivienda_Arriendo = ifelse(P5090 == 3, 1, 0),
      Vivienda_Usufructo = ifelse(P5090 == 4, 1, 0),
      Vivienda_Posesion_Sin_Titulo = ifelse(P5090 == 5, 1, 0),
      Vivienda_Otra = ifelse(P5090 == 6, 1, 0),
      Indicador_Arriendo = ifelse(P5090 == 3, 1, 0)  # Indicador si el hogar está en arriendo
    )
  
  # Número de personas estudiando en el hogar (P6240 == 3)
  num_personas_estudiando <- df_personas %>%
    group_by(id) %>%
    summarise(Num_Personas_Estudiando = sum(P6240 == 3, na.rm = TRUE))
  
  # Proporción de personas estudiando en el hogar (P6240 == 3)
  proporcion_estudiando <- df_personas %>%
    group_by(id) %>%
    summarise(Proporción_Estudiando = mean(P6240 == 3, na.rm = TRUE))
  
  # Número de dependientes en el hogar (P6040 menor a 15 o mayor a 65)
  num_dependientes <- df_personas %>%
    group_by(id) %>%
    summarise(Num_Dependientes = sum(P6040 < 15 | P6040 > 65, na.rm = TRUE))
  
  # Proporción de dependientes en el hogar (P6040 menor a 15 o mayor a 65)
  dependientes_hogar <- df_personas %>%
    group_by(id) %>%
    summarise(Proporción_Dependientes = mean(P6040 < 15 | P6040 > 65, na.rm = TRUE))
  
  # Número de mujeres en el hogar (P6020 == 2)
  num_mujeres <- df_personas %>%
    group_by(id) %>%
    summarise(Num_Mujeres = sum(P6020 == 2, na.rm = TRUE))
  
  # Proporción de mujeres en el hogar (P6020 == 2)
  mujeres_hogar <- df_personas %>%
    group_by(id) %>%
    summarise(Proporción_Mujeres = mean(P6020 == 2, na.rm = TRUE))
  
  # Nivel educativo categorizado y máximo por hogar
  nivel_educativo_maximo <- df_personas %>%
    group_by(id) %>%
    summarise(Nivel_Educativo_Maximo = max(P6210, na.rm = TRUE))
  
  # Dummies para nivel educativo máximo
  nivel_educativo_dummies <- nivel_educativo_maximo %>%
    mutate(
      Max_Ninguno = ifelse(Nivel_Educativo_Maximo == 1, 1, 0),
      Max_Preescolar = ifelse(Nivel_Educativo_Maximo == 2, 1, 0),
      Max_Basica_Primaria = ifelse(Nivel_Educativo_Maximo == 3, 1, 0),
      Max_Basica_Secundaria = ifelse(Nivel_Educativo_Maximo == 4, 1, 0),
      Max_Media = ifelse(Nivel_Educativo_Maximo == 5, 1, 0),
      Max_Superior = ifelse(Nivel_Educativo_Maximo == 6, 1, 0)
    )
  
  # Número de personas con nivel educativo de media o superior (excluyendo 9)
  num_media_superior <- df_personas %>%
    group_by(id) %>%
    summarise(Num_Personas_Media_Superior = sum(ifelse(P6210 >= 5 & P6210 != 9, 1, 0), na.rm = TRUE))
  
  # Indicador de hogar con jefe femenino
  jefe_femenino <- df_personas %>%
    filter(Orden == 1) %>%
    mutate(Indicador_Hogar_Jefe_Femenino = ifelse(P6020 == 2, 1, 0)) %>%
    select(id, Indicador_Hogar_Jefe_Femenino)
  
  # Indicador de hacinamiento crítico (Nper / P5010 > 3)
  if ("P5010" %in% colnames(df_hogares) & "Nper" %in% colnames(df_hogares)) {
    df_hogares <- df_hogares %>%
      mutate(Indicador_Hacinamiento_Critico = ifelse(Nper / P5010 > 3, 1, 0))
  } else {
    df_hogares$Indicador_Hacinamiento_Critico <- NA
  }
  
  # Unir todas las variables calculadas a la base de hogares
  df_hogares <- df_hogares %>%
    left_join(num_personas_trabajando, by = "id") %>%
    left_join(ocupados_hogar, by = "id") %>%
    left_join(promedio_edad_ocupados, by = "id") %>%
    left_join(num_personas_estudiando, by = "id") %>%
    left_join(proporcion_estudiando, by = "id") %>%
    left_join(num_dependientes, by = "id") %>%
    left_join(dependientes_hogar, by = "id") %>%
    left_join(num_mujeres, by = "id") %>%
    left_join(mujeres_hogar, by = "id") %>%
    left_join(nivel_educativo_dummies, by = "id") %>%
    left_join(num_media_superior, by = "id") %>%
    left_join(jefe_femenino, by = "id")
  
  # Retornar la base de datos final con las variables agregadas
  return(df_hogares)
}


# Aplicar la función a las bases de train y test
train_hogares_final <- agregar_variables_hogar(train_personas, train_hogares)
test_hogares_final <- agregar_variables_hogar(test_personas, test_hogares)


# Guardar las bases finales
write.csv(train_hogares_final, "train_hogares_final.csv", row.names = FALSE)
write.csv(test_hogares_final, "test_hogares_final.csv", row.names = FALSE)

# Imprimir una muestra de las nuevas variables en test_hogares_final
head(test_hogares_final)

