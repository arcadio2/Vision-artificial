#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <stdlib.h> 
#define PI 3.141592653589

using namespace cv;
using namespace std;


/*Funcion para cargar la imagen con opencv*/
Mat obtenerImagen(char NombreImagen[]) {

	Mat imagen; // Matriz que contiene nuestra imagen sin importar el formato
	

	/*********Lectura de la imagen*********/
	imagen = imread(NombreImagen);

	if (!imagen.data){
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	return imagen; 
}
void mostrarImagen(Mat imagen) {
	/************Procesos*********/
	int fila_original = imagen.rows;
	int columna_original = imagen.cols;//Lectur de cuantas columnas

	cout << "filas: " << fila_original << endl;
	cout << "columnas: " << columna_original << endl;

	namedWindow("Práctica 3", WINDOW_AUTOSIZE);//Creación de una ventana
	imshow("Práctica 3", imagen);

	/************************/

	waitKey(0); //Función para esperar
	
}
void mostrarImagen2(Mat imagen) {

	namedWindow("practica3", WINDOW_AUTOSIZE);
	imshow("practica3", imagen);
}

/*FUncion que genera el kernel gaussiano de manera dinamica*/
double ** generacionKernel(double sigma, int kernel) {
	//Mat matriz; 
	Mat matriz(kernel, kernel, CV_8UC1);

	//empezamos desde el centro
	int centro = (kernel - 1) / 2; 

	//creamos la matriz para el kernel
	double ** matrizprueba = new double*[kernel];
	double ** matrizFinal = new double* [kernel];

	double suma = 0; 

	//recorrido para llenar la matriz
	for (int i = 0;i < kernel;i++) {
		matrizprueba[i] = new double[kernel]; 
		for (int j = 0; j < kernel; j++){
			//como iniciamos desde el centro, para iniciar de arriba abajo hacemos las
			//siguientes operaciones
			int posx = i - centro; 
			int posy = (j - centro)*-1; 
		
			double valor = 1 / (2 * PI * sigma * sigma); 
			valor = valor * exp(-(pow(posx, 2) + pow(posy, 2)) / (2 * pow(sigma, 2)) ); 
			suma += valor; 
			matrizprueba[i][j] = valor;

			//matriz.at<uchar>(Point(posy, posx)) = uchar(valor);

			cout << valor << "\t"; 

		}
		cout << "\n"; 
	}
	cout << "Suma " <<suma << "\n";
	cout << "---------------------------------------------------------------\n";
	//normalizacion del kernel gaussiano, con respecto a la suma
	for (int i = 0;i < kernel;i++) {
		matrizFinal[i] = new double[kernel];
		for (int j = 0; j < kernel; j++) {
			matrizFinal[i][j] = matrizprueba[i][j]/suma;
			cout << matrizFinal[i][j] << "\t";
		}
		cout << "\n";
	}

	cout << "---------------------------------------------------------------\n";
	return matrizFinal; 
}

/*Función que pasa el kernel por la imagen y hace la convolución*/
double operacionConvolucion(int i, int j, int kernel, int exceso, Mat imagen, double** matriz) {
	double resultado = 0;
	/*Hacemos el filtro*/;
	//recorremos la matriz

	
	for (int k = 0; k < kernel; k++) {
		for (int l = 0; l < kernel; l++) {

			//valor de la imagen con bordes
			int valorImagen = imagen.at<uchar>(Point(j - exceso + l, i - exceso + k));

			//se multplica el valor del kernel por el de la imagen, tomando el centro
			//como el punto en el que estamos, y se hacen las operaciones para que
			//vayan correspondiente los pixeles de la imagen a los del kernel
			resultado += (matriz[k][l] * valorImagen);

			//resultado += (matriz[k][l]);
		}
	}
	
	//resultado = resultado / pow(kernel, 2);

	resultado = static_cast<int>(resultado);
	return resultado;
}

/*Función que recorre toda la imagen para pasar a la convolución*/
Mat convolucion(Mat imagen, int kernel, double** matriz) {
	//imagen chida con bordes
	int rows = imagen.rows;
	int cols = imagen.cols;
	cout << rows << "x" << cols << "\n";
	int exceso = (kernel - 1) / 2;

	cout << "exceso " << exceso << "\n";
	Mat filtrada(rows-exceso*2, cols-exceso*2, CV_8UC1);

	//mostrarImagen(imagen);
	//rows = filtrada.rows;
	//cols = filtrada.cols;
	//printf("xd %d\n", filtrada.cols);
	//recorrenis la imagen con bordes
	for (int i = 0 ; i < rows; i++) {
		
		//quita los bordes de izquierda y derecga
		if ((i < (rows -exceso) && i >= exceso)) {

			for (int j = 0; j < cols; j++) {
				//quita los bordes de arriba y abajo
				if ((j < (cols -exceso ) && j >= exceso)) {
					///
					int resultado = operacionConvolucion(i, j, kernel, exceso, imagen, matriz);
					

					//resultado = resultado / pow(kernel, 2);

					resultado = abs(static_cast<int>(resultado));

					filtrada.at<uchar>(Point(j - exceso, i - exceso)) = uchar(resultado);
				}
			}
		}
		
	
	}

	
	
	
	//mostrarImagen(filtrada);
	//waitKey(0);
	return filtrada;
	
}


/*Esta función genera un borde a la imagen, con respecto al exceso que nos de el kernel*/
Mat bordearImagen(Mat imagen,int kernel) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	int exceso = (kernel - 1) / 2; //exceso que sobresale de los bordes

	Mat grises(rows , cols , CV_8UC1);
	//la imagen con el exceso multiplicado x2 por izquierda derecha, arriba y abajo
	Mat grande(rows + exceso*2, cols + exceso*2, CV_8UC1);

	Mat filtrada(rows, cols, CV_8UC1);
	Mat ImagenFiltrada;

	printf(" A %d \n", grande.rows);
	//GaussianBlur(imagen, ImagenFiltrada, Size(3, 3), 0, 0, 0);

	//si la función está en escala de grises no pasa nada
	if (imagen.type() == CV_8UC1) {
		grises = imagen; 
	}
	else { //si no esta en escala de grises, la pasamos a escala de grises
		cvtColor(imagen, grises, COLOR_BGR2GRAY);
	}
	

	double rojo, azul, verde, gris_p;

	//recorremos hasta el exceso que tendremos de nuestra imagen bordeada
	for (int i = 0; i < rows + exceso*2 ; i++) {
		for (int j = 0; j < cols + exceso*2; j++) {
			
			//si sobre pasa los limites de la imagen le asignamos un negro como borde
			if ( i > rows + exceso - 1 || i < exceso) { 
				grande.at<uchar>(Point(j, i)) = uchar(0);
				//cout << "entra\n";
			

			}else if (j > cols + exceso - 1 || j < exceso) { 
				//cout << "j " << j << "\n";
				grande.at<uchar>(Point(j, i)) = uchar(0);
				//cout << "entra\n";
			}else {
				/*valores de la imagen original*/
													//cinversion de coordenadas
				/*azul = imagen.at<Vec3b>(Point(j - exceso, i - exceso)).val[0];
				verde = imagen.at<Vec3b>(Point(j - exceso, i-exceso)).val[1];
				rojo = imagen.at<Vec3b>(Point(j - exceso, i - exceso)).val[2];*/

				gris_p = grises.at<uchar>(Point(j - exceso, i - exceso));
				//el valor de gris promediado lo obtenemos sumando cada valor de 
				//rojo, verde y azul sobre 3
				//gris_p = (azul + verde + rojo) / 3;

				grande.at<uchar>(Point(j, i)) = uchar(gris_p);

				
			}
			
		
		}
	}
	imwrite("borde.png", grande); 
	
	//mostrarImagen(filtrada);
	//mostrarImagen(ImagenFiltrada);

	return grande; 

}

/*
Función que ecualiza la imagen, esta función normaliza el histograma de la imagen
Haciendo uso del maximo y el minimo valor que sea diferente a 0
*/
Mat ecualizacion(Mat imagen) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	Mat ecualizada(rows, cols, CV_8UC1);

	//arreglo para obtener los valores de la imagen en 1 dimensión
	double* flateada = new double [rows*cols];
	//sumamos la 1 con la anterior

	double suma = 0; 
	int c = 0; 
	//la creamos en arreglo
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){

			suma += imagen.at<uchar>(Point(j, i)); 
			flateada[c] = imagen.at<uchar>(Point(j, i));
			c++; 
		}
	}
	//obtener el maximo y minimo
	double max=0, min=10000;
	for (int i = 0; i < rows*cols; i++){
		//obteniendo el minimo diferente a 0
		if (flateada[i] < min && flateada[i]!=0) {
			min = flateada[i];
		}
		if (flateada[i] > max) {
			max = flateada[i];
		}
	}
	//(cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	//usamos una formula para normalizar la imagen
	c = 0; 
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			double valor = (flateada[c] - min) * 255 / (max - min); 
			valor = static_cast<int>(valor);
			ecualizada.at<uchar>(Point(j, i)) = uchar(valor);
			c++; 
		}
	}


	return ecualizada; 
}

/*
Esta función genera el filtro de sobel con respecto a la magnitud
el argumento de absoluto, si es true usa la formula |gx| + |gy|, sino usa la raiz

*/
Mat sobelXY(Mat x, Mat y, bool absoluto=false) {
	int rows = x.rows;
	int cols = y.cols;
	

	Mat filtro_total(rows, cols, CV_8UC1);

	//Mat prueba(rows, cols, CV_8UC1);

	double magnitud, direccion; 
	double valor_x, valor_y; 
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){

			valor_x = x.at<uchar>(Point(j, i));
			valor_y = y.at<uchar>(Point(j, i));
			if (absoluto) {
				magnitud = abs(valor_x) + abs(valor_y);
			}else { //si el argumento de absoluto es falso usa la raiz
				magnitud = sqrt(pow(valor_x, 2) + pow(valor_y, 2));		
			}
			magnitud = static_cast<int>(magnitud); 
			/*if (magnitud > umbral) {
				magnitud = 255; 
			}else {
				magnitud = 0; 
			}*/
			//asignamos la magnutd a cada punto
			filtro_total.at<uchar>(Point(j, i)) = uchar(magnitud); 
			direccion = atan(valor_y/valor_x);
			//prueba.at<uchar>(Point(j, i)) = uchar(direccion);

		}
	}

	//mostrarImagen2(prueba);
	return filtro_total; 
}

/*
Esta función busca la dirección a la que va el pixel, es decir hacia que vecino va,
hace el non-max
*/
Mat canny(Mat x, Mat y) {
	int rows = x.rows;
	int cols = x.cols;
	Mat vecinos(rows, cols, CV_8UC1);

	double  direccion;
	double valor_x, valor_y, valor_x_a, valor_x_s, valor_y_a, valor_y_s;
	double magnitud, magnitud_anterior, magnitud_siguiente;
	//recorremos la imagen, quitando un pixel de cada lado para tener consistencia en 
	//las operaciones
	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			//valores actuales
			valor_x = x.at<uchar>(Point(j, i));
			valor_y = y.at<uchar>(Point(j, i));
			magnitud = sqrt(pow(valor_x, 2) + pow(valor_y, 2));

			vecinos.at<uchar>(Point(j, i)) = uchar(static_cast<int>(0));

			direccion = valor_y / valor_x;
			//izquierda y derecha
			if (direccion <= tan(22.5) && direccion > tan(-22.5)) {
				//valores anteriores
				valor_x_a = x.at<uchar>(Point(j - 1, i));
				valor_y_a = y.at<uchar>(Point(j - 1, i));
				//valores siguientes
				valor_x_s = x.at<uchar>(Point(j + 1, i));
				valor_y_s = y.at<uchar>(Point(j + 1, i));


				magnitud_anterior = sqrt(pow(valor_x_a, 2) + pow(valor_y_a, 2));
				magnitud_siguiente = sqrt(pow(valor_x_s, 2) + pow(valor_y_s, 2));

				if (magnitud > magnitud_anterior && magnitud > magnitud_siguiente) {

					vecinos.at<uchar>(Point(j, i)) = uchar(static_cast<int>(magnitud));

				}

			}
			else if (direccion <= tan(67.5) && direccion > tan(22.5)) {
				//diagonal izquierda-abajo, diagonal dercha-arriba
				//valores anteriores
				valor_x_a = x.at<uchar>(Point(j - 1, i - 1));
				valor_y_a = y.at<uchar>(Point(j - 1, i - 1));
				//valores siguientes
				valor_x_s = x.at<uchar>(Point(j + 1, i + 1));
				valor_y_s = y.at<uchar>(Point(j + 1, i + 1));


				magnitud_anterior = sqrt(pow(valor_x_a, 2) + pow(valor_y_a, 2));
				magnitud_siguiente = sqrt(pow(valor_x_s, 2) + pow(valor_y_s, 2));
				if (magnitud > magnitud_anterior && magnitud > magnitud_siguiente) {
					vecinos.at<uchar>(Point(j, i)) = uchar(static_cast<int>(magnitud));

				}
			}
			else if (direccion <= tan(-22.5) && direccion > tan(-67.5)) {//diagonal 
				//diagonal derecha-abajo, diagonal izquierda-arriba
				//valores anteriores
				valor_x_a = x.at<uchar>(Point(j - 1, i + 1));
				valor_y_a = y.at<uchar>(Point(j - 1, i + 1));
				//valores siguientes
				valor_x_s = x.at<uchar>(Point(j + 1, i - 1));
				valor_y_s = y.at<uchar>(Point(j + 1, i - 1));


				magnitud_anterior = sqrt(pow(valor_x_a, 2) + pow(valor_y_a, 2));
				magnitud_siguiente = sqrt(pow(valor_x_s, 2) + pow(valor_y_s, 2));
				if (magnitud > magnitud_anterior && magnitud > magnitud_siguiente) {
					vecinos.at<uchar>(Point(j, i)) = uchar(static_cast<int>(magnitud));

				}
			}
			else {
				//valores anteriores
				valor_x_a = x.at<uchar>(Point(j - 1, i + 1));
				valor_y_a = y.at<uchar>(Point(j - 1, i + 1));
				//valores siguientes
				valor_x_s = x.at<uchar>(Point(j, i + 1));
				valor_y_s = y.at<uchar>(Point(j, i + 1));


				magnitud_anterior = sqrt(pow(valor_x_a, 2) + pow(valor_y_a, 2));
				magnitud_siguiente = sqrt(pow(valor_x_s, 2) + pow(valor_y_s, 2));
				if (magnitud > magnitud_anterior && magnitud > magnitud_siguiente) {
					vecinos.at<uchar>(Point(j, i)) = uchar(static_cast<int>(magnitud));

				}
			}
		}
	}
	return vecinos;
}


/*Función que crea el filtro sobel y el filtro canny
Dentro de esta se inicializan los valores de los kernel para gx y gy
*/
Mat * filtroSobel(Mat imagenGaus) {

	double ** kernel_x = new double* [3];
	double ** kernel_y = new double* [3];
	for (int i = 0;i < 3;i++) {
		kernel_x[i] = new double[3];
		kernel_y[i] = new double[3];
	}
	//rellenamos los kernels
	kernel_x[0][0] = -1;
	kernel_x[0][1] = 0;
	kernel_x[0][2] = 1;

	kernel_x[1][0] = -2;
	kernel_x[1][1] = 0;
	kernel_x[1][2] = 2;

	kernel_x[2][0] = -1;
	kernel_x[2][1] = 0;
	kernel_x[2][2] = 1;
	//y
	kernel_y[0][0] = -1;
	kernel_y[0][1] = -2;
	kernel_y[0][2] = -1;

	kernel_y[1][0] = 0;
	kernel_y[1][1] = 0;
	kernel_y[1][2] = 0;

	kernel_y[2][0] = 1;
	kernel_y[2][1] = 2;
	kernel_y[2][2] = 1;

	//obtenemos la bordeada para poder hacer la convolución
	Mat bordeada = bordearImagen(imagenGaus, 3);

	//obtenemos los resultados al convolucionar las imagenes
	Mat gx = convolucion(bordeada, 3, kernel_x);
	Mat gy = convolucion(bordeada, 3, kernel_x);

	Mat filtro_total = sobelXY(gx,gy);

	Mat otra = canny(gx, gy); 


	//mostrarImagen(gx); 
	//mostrarImagen(gy);

	//mostrarImagen(filtro_total);
	//mostrarImagen(otra);
	//creamos un arreglo para regresar las 4 imagenes
	Mat* imagenes = new Mat[4];
	imagenes[0] = gx; 
	imagenes[1] = gy;
	imagenes[2] = filtro_total;
	imagenes[3] = otra; 

	return imagenes; 


}

/*
Esta función obtiene la imagen ya lista para umbralar alto o bajo, estos valores
son porcentajes, el alto del maximo y el bajo del umbral alto
*/
Mat umbralado(Mat imagen, double umbral_alto, double umbral_bajo) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	Mat umbralada(rows, cols, CV_8UC1);

	double maximo = 0; 
	double valor; 
	//obtenemos el maximo
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			valor = imagen.at<uchar>(Point(j, i)); 
			if (maximo < valor) {
				maximo = valor; 
			}
		}
	}
	//sacamos el umbral alto con respecto al maximo
	umbral_alto = static_cast<int>( (umbral_alto * maximo) / 100 );
	//sacamos el umbral bajo con respecto al umbral alto
	umbral_bajo = static_cast<int>( (umbral_alto * umbral_bajo) / 100);


	//los balores menores al umbral los despreciamos
	//los valores entre el bajo y el alto los mantenemos
	//los valores mayores al umbral alto los hacemos 255
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			valor = imagen.at<uchar>(Point(j, i));
			if (valor <= umbral_bajo) {
				umbralada.at<uchar>(Point(j, i)) = uchar(0);
			}
			else if (valor < umbral_alto) {
				umbralada.at<uchar>(Point(j, i)) = uchar(valor);
			}
			else {
				umbralada.at<uchar>(Point(j, i)) = uchar(255);
			}
		}
	}



	return umbralada; 
}

/*Esta funcion le hace un umbral a la imagen
Pero solo en el caso en el que alguno de sus vecinos tenga un pixel fuerte
*/
Mat umbral_final(Mat imagen, int umbral) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	Mat umbralada(rows, cols, CV_8UC1);
	//umbral mayor a 130


	double vecino_i, vecino_d, vecino_ar, vecino_ab;
	double vecino_i_ar, vecino_d_ar, vecino_i_ab, vecino_d_ab;

	double valor;
	for (int i = 1; i < rows - 1; i++) {
		for (int j = 1; j < cols - 1; j++) {
			vecino_i  = imagen.at<uchar>(Point(j, i-1)); //izquierda
			vecino_d  = imagen.at<uchar>(Point(j, i + 1)); //derecha
			vecino_ar = imagen.at<uchar>(Point(j + 1, i)); //arriba
			vecino_ab = imagen.at<uchar>(Point(j - 1, i)); //abajo
	
			vecino_i_ar = imagen.at<uchar>(Point(j + 1, i - 1)); //izquierda arriba
			vecino_d_ar = imagen.at<uchar>(Point(j + 1 , i + 1 )); // derecha arriba
			vecino_i_ab = imagen.at<uchar>(Point(j - 1, i - 1 )); //izquierda abajo
			vecino_d_ab = imagen.at<uchar>(Point(j - 1, i + 1 )); //derecha abajo

			//si alguno es mayor al umbral que dimos, mantenemos el valor y sino lo despreciamos
			if (vecino_i >= umbral || vecino_d >= umbral || vecino_ar >= umbral 
				|| vecino_ab >= umbral|| vecino_i_ar >= umbral 
				||  vecino_d_ar >= umbral || vecino_i_ab >= umbral 
				||  vecino_d_ab >= umbral) {
				//en 255 o el valor
				umbralada.at<uchar>(Point(j, i)) = imagen.at<uchar>(Point(j, i));
			}
			else {
				umbralada.at<uchar>(Point(j, i )) =uchar(0);
			}
		}
	}
	return umbralada;
}

/*Función main, aqui se obtienen las imagenes y se muestran*/
int main() {

	char NombreImagen[] = "lena.png";
	Mat imagen = obtenerImagen(NombreImagen);
	//mostrarImagen(imagen);

	double sigma;

	cout << "Elige el valor de sigma: ";
	cin >> sigma;

	int kernel;
	cout << "elige un tamaño para kernel: ";

	cin >> kernel;
	while (kernel < 1 || kernel % 2 == 0) {
		cout << "Error! Elige in kernel de tamaño impar ";

		cin >> kernel;
	}


	double** matriz_kernel = generacionKernel(sigma, kernel);

	Mat bordeada = bordearImagen(imagen, kernel);

	//mostrarImagen(bordeada); 

	//imagen con filtro gaussiano
	Mat filtrada = convolucion(bordeada, kernel, matriz_kernel);

	//Mat sobel = filtroSobel(filtrada);
	//imagen ecualizada 
	Mat ecualizada = ecualizacion(filtrada); 
	//imagenes obtenidas del sobel mas la del non-max
	Mat *  sobel_imgs = filtroSobel(ecualizada);
	/*Mostrar las imagenes*/

	Mat gx = sobel_imgs[0]; 
	Mat gy = sobel_imgs[1];
	Mat sobel = sobel_imgs[2];
	Mat canny_img = sobel_imgs[3];

	Mat umbralada = umbralado(sobel, 50, 30); //canny o sobel

	Mat umbralada_final = umbral_final(umbralada, 250);


	Mat umbralada_canny = umbralado(canny_img, 50, 30); //canny o sobel

	Mat umbralada_final_canny = umbral_final(umbralada_canny, 240);

	//umbralada_final = umbral_final(umbralada_final, 200);

	imshow("original", imagen);
	printf("Original: %dx%d\n", imagen.cols, imagen.rows);

	imshow("Con bordes", bordeada);
	printf("Con bordes: %dx%d\n", bordeada.cols, bordeada.rows);


	imshow("Gaussiano", filtrada);
	printf("Gaussiano: %dx%d\n", filtrada.cols, filtrada.rows);

	imshow("Ecualizada", ecualizada);
	printf("Ecualizada: %dx%d\n", ecualizada.cols, ecualizada.rows);

	waitKey(0); 
	imshow("Gx", gx);
	printf("Gx: %dx%d\n", gx.cols, gx.rows);

	imshow("Gy", gy);
	printf("Gy: %dx%d\n", gy.cols, gy.rows);

	imshow("Sobel |G|", sobel);
	printf("|G|: %dx%d\n", sobel.cols, sobel.rows);

	imshow("Canny", canny_img);
	printf("Canny: %dx%d\n", canny_img.cols, canny_img.rows);

	waitKey(0);

	imshow("Umbralada sobel", umbralada);
	printf("Umbralada sobel: %dx%d\n", umbralada.cols, umbralada.rows);

	imshow("Umbralada final sobel", umbralada_final);
	printf("Umbralada final sobel: %dx%d\n", umbralada_final.cols, umbralada_final.rows);

	imshow("Umbralada con canny", umbralada_canny);
	printf("Umbralada con canny: %dx%d\n", umbralada_canny.cols, umbralada_canny.rows);

	imshow("Umbralada final canny", umbralada_final_canny);
	printf("Umbralada final canny: %dx%d\n", umbralada_final_canny.cols, umbralada_final_canny.rows);

	waitKey(0);
	
	return 1;
}


