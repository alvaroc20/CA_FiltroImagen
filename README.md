# CA_FiltroImagen  

### Manual de Usuario  
make compile: **compilar** el archivo de CUDA.  
make run: **ejecutar** el archivo que se ha generado después de compilar.  
make cleanPhoto: Elimina cualquier imagen que haya producido el programa.  
make cleanExe: Elimiana el exe del programa.  
make clean: Elimina cualquier imagen y exe que el programa haya generado.  


Para cambiar el **número de hilos** que se están utilizando o la **ruta** de la imagen que quieres introducir, hágalo desde el Makefile.  

Es posible que se necesite alguna librería externa, puede instalarla de la siguiente manera:  
    sudo apt-get update  
    sudo apt-get upgrade  
    
    sudo apt-get install libopencv-dev  
    sudo apt-get install pkg-config  