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
double ** generacionKernel(double sigma, int kernel) {
	//Mat matriz; 
	Mat matriz(kernel, kernel, CV_8UC1);
	int centro = (kernel - 1) / 2; 

	double ** matrizprueba = new double*[kernel];
	double ** matrizFinal = new double* [kernel];

	double suma = 0; 

	for (int i = 0;i < kernel;i++) {
		matrizprueba[i] = new double[kernel];
		for (int j = 0; j < kernel; j++){
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
	//normalizacion
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

double operacionConvolucion(int i, int j, int kernel, int exceso, Mat imagen, double** matriz) {
	double resultado = 0;
	/*Hacemos el filtro*/;
	//recorremos la matriz
	for (int k = 0; k < kernel; k++) {
		for (int l = 0; l < kernel; l++) {

			//valor de la imagen con bordes
			int valorImagen = imagen.at<uchar>(Point(j - exceso + l, i - exceso + k));

			resultado += (matriz[k][l] * valorImagen);

			//resultado += (matriz[k][l]);
		}
	}
	
	//resultado = resultado / pow(kernel, 2);

	resultado = static_cast<int>(resultado);
	return resultado;
}

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
					//int resultado = operacionConvolucion(i, j, kernel, exceso, imagen, matriz);
					double resultado = 0;
					/*Hacemos el filtro*/;
					//recorremos la matriz
					for (int k = 0; k < kernel; k++) {
						for (int l = 0; l < kernel; l++) {


							//valor de la imagen con bordes
							int valorImagen = imagen.at<uchar>(Point(j - exceso + l, i - exceso + k));

							resultado += (matriz[k][l] * valorImagen);

							//resultado += (matriz[k][l]);
						}
					}

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



Mat bordearImagen(Mat imagen,int kernel) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	int exceso = (kernel - 1) / 2;

	Mat grises(rows , cols , CV_8UC1);
	Mat grande(rows + exceso*2, cols + exceso*2, CV_8UC1);

	Mat filtrada(rows, cols, CV_8UC1);
	Mat ImagenFiltrada;

	printf(" A %d \n", grande.rows);
	//GaussianBlur(imagen, ImagenFiltrada, Size(3, 3), 0, 0, 0);

	//convierte a escala de grises
	if (imagen.type() == CV_8UC1) {
		grises = imagen; 
	}
	else {
		cvtColor(imagen, grises, COLOR_BGR2GRAY);
	}
	

	double rojo, azul, verde, gris_p;

	for (int i = 0; i < rows + exceso*2 ; i++) {
		for (int j = 0; j < cols + exceso*2; j++) {
			
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
Mat ecualizacion(Mat imagen) {
	int rows = imagen.rows;
	int cols = imagen.cols;
	Mat ecualizada(rows, cols, CV_8UC1);


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
		if (flateada[i] < min && flateada[i]!=0) {
			min = flateada[i];
		}
		if (flateada[i] > max) {
			max = flateada[i];
		}
	}
	//(cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	c = 0; 
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			ecualizada.at<uchar>(Point(j, i)) = (flateada[c]-min)*255/(max-min);
			c++; 
		}
	}


	return ecualizada; 
}

Mat sobelXY(Mat x, Mat y, bool absoluto=false) {
	int rows = x.rows;
	int cols = y.cols;
	

	Mat filtro_total(rows, cols, CV_8UC1);

	//Mat prueba(rows, cols, CV_8UC1);
	
	int umbral = 120; 
	int umbral_bajo = 5; 
	double magnitud, direccion; 
	double valor_x, valor_y; 
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){

			valor_x = x.at<uchar>(Point(j, i));
			valor_y = y.at<uchar>(Point(j, i));
			if (absoluto) {
				magnitud = abs(valor_x) + abs(valor_y);
			}else {
				magnitud = sqrt(pow(valor_x, 2) + pow(valor_y, 2));		
			}
			magnitud = static_cast<int>(magnitud); 
			if (magnitud > umbral) {
				magnitud = 255; 
			}else {
				magnitud = 0; 
			}

			filtro_total.at<uchar>(Point(j, i)) = uchar(magnitud); 
			direccion = atan(valor_y/valor_x);
			//prueba.at<uchar>(Point(j, i)) = uchar(direccion);

		}
	}

	//mostrarImagen2(prueba);
	return filtro_total; 
}

Mat canny(Mat x, Mat y) {
	int rows = x.rows;
	int cols = x.cols;
	Mat vecinos(rows, cols, CV_8UC1);

	double  direccion;
	double valor_x, valor_y, valor_x_a, valor_x_s, valor_y_a, valor_y_s;
	double magnitud, magnitud_anterior, magnitud_siguiente;
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

Mat filtroSobel(Mat imagenGaus) {

	double ** kernel_x = new double* [3];
	double ** kernel_y = new double* [3];
	for (int i = 0;i < 3;i++) {
		kernel_x[i] = new double[3];
		kernel_y[i] = new double[3];
	}
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

	//obtenemos la bordeada
	Mat bordeada = bordearImagen(imagenGaus, 3);

	Mat gx = convolucion(bordeada, 3, kernel_x);
	Mat gy = convolucion(bordeada, 3, kernel_x);

	Mat filtro_total = sobelXY(gx,gy);

	Mat otra = canny(gx, gy); 


	mostrarImagen(gx); 
	mostrarImagen(gy);

	mostrarImagen(filtro_total);
	mostrarImagen(otra);


	return bordeada; 


}


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

	Mat ecualizada = ecualizacion(filtrada); 

	Mat sobel = filtroSobel(ecualizada);
	/*Mostrar las imagenes*/

	imshow("original", imagen);
	printf("Original: %dx%d\n", imagen.cols, imagen.rows);

	imshow("Con bordes", bordeada);
	printf("Con bordes: %dx%d\n", bordeada.cols, bordeada.rows);


	imshow("Gaussiano", filtrada);
	printf("Gaussiano: %dx%d\n", filtrada.cols, filtrada.rows);

	imshow("Ecualizada", ecualizada);
	printf("Ecualizada: %dx%d\n", ecualizada.cols, ecualizada.rows);

	waitKey(0); 
	
	return 1;
}


