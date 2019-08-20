/*
 * MatchGen.hpp
 *
 *  Created on: Mar 18, 2019
 *      Author: blakemoore
 */

#ifndef MATCHGEN_HPP_
#define MATCHGEN_HPP_

#include <fftw3.h>


//Computes the SNR squared for a complex type
double get_snr(vector<complex<double>> &vect, vector<double> &noise){
	double sum = 0;
	int N = vect.size();
	for(int i = 0; i < N; i++){
		sum += 4./exp(noise[i])*(vect[i].real()*vect[i].real() + vect[i].imag()*vect[i].imag());
	}
	return sum;
}
//Compute the SNR squared for a vector where the real part is vect[i][1] and imaginary part is vect[i][2]
double get_snr(vector<vector<double>> &vect, vector<double> &noise){
	double sum = 0;
	int N = vect.size();
	for(int i = 0; i < N; i++){
		sum += 4./exp(noise[i])*(vect[i][1]*vect[i][1] + vect[i][2]*vect[i][2]);
	}
	return sum;
}
// computes the inverse fourier transform of the integrand (i.e. the integrand in the match for many different time offsets)
fftw_complex* take_invft_gj(vector<vector<double>> &t4, vector<complex<double>> &f2, vector<double> &noise){
	// prepare the inner product integrand
		fftw_complex *in, *out;
		int N = f2.size();
	    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
	    for(int i = 0; i < N; i++){
	    	in[i][0] = 4./exp(noise[i])*(f2[i].real()*t4[i][1] + f2[i].imag()*t4[i][2]); //real part
	    	in[i][1] = 4./exp(noise[i])*(f2[i].real()*t4[i][2] - f2[i].imag()*t4[i][1]); //imaginary part
	    //	in[i][1] = 4./exp(noise[i])*(f2[i].imag()*t4[i][1] - f2[i].real()*t4[i][2]);
	    }

		//routine to get the inner product maximized over a time and phase shift
	    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	    fftw_execute(p);
	    fftw_destroy_plan(p);

	    fftw_free(in);
	    return out;
}
// Takes the absolute value of the IFT of the integrand of match
vector<double> abs_gj(vector<vector<double>> t4, vector<complex<double>> f2, vector<double> noise){
	fftw_complex *out;
	int N = f2.size();
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = take_invft_gj(t4, f2, noise);
    vector<double> absvals(N);
    for (int i = 0; i < N; i++){
    	absvals[i] = pow(out[i][0]*out[i][0] + out[i][1]*out[i][1], 1./2.);
    }
    fftw_free(out);
    return absvals;
}

// Computes the match between a signal "t4h" which is assumed to not be harmonically decomposed, and "harms" which is assumed to be harmonically decomposed
// the fact that <complex<double>> is not consistently used throughout is unfortunate
double match_full_2(vector<vector<double> > t4h, vector<vector<complex<double>>> harms, gsl_spline *noise_interp, gsl_interp_accel *acc){

	int lenth = t4h.size();
	vector<double> freqs(lenth);
	for(int i = 0; i < lenth; i++){
		freqs[i] = t4h[i][0];
	}
	int size = harms.size();

	// get the noise
	vector<double> noise(lenth);
	for (int i = 0; i < lenth; i++)	{
		if (freqs[i] > 1 &&  freqs[i] < 4096 ) {
		noise[i] = gsl_spline_eval (noise_interp, freqs[i], acc);
		} else {
			noise[i] = pow(10,10);
		}
	}

//	cout << "here" << endl;
	// compute the sum of abs
	vector<vector<double>> hold_abs(size, vector<double>(lenth));
	for (int i = 0; i < size; i++){
		hold_abs[i] = abs_gj(t4h, harms[i], noise);
	}
//	cout << "here" << endl;
	vector<double> sum_abs(lenth);
	for (int i = 0; i < lenth; i++){
		for(int j = 0; j < size; j++){
			sum_abs[i] += hold_abs[j][i];
		}
	}
	double maxprod = *max_element(sum_abs.begin(), sum_abs.end());

	//compute t4snr
	double t4snr = get_snr(t4h, noise);

	//compute f2snr
	vector<complex<double>> sumf2 (lenth);
	complex<double> tmpsum = 0;
	for(int j = 0; j < lenth; j++){
		for(int i = 0; i < size; i++){
			tmpsum += harms[i][j];
		}
		sumf2[j] = tmpsum;
		tmpsum = 0;
	}
	double f2snr = get_snr(sumf2, noise);
//
//	cout << "f2snr = " << f2snr*freqs[1] << endl;
//	cout << "t4snr = " << t4snr*freqs[1] << endl;
//	cout << "inner prod (max'd) = " << maxprod*freqs[1] << endl;


	return maxprod/pow(t4snr*f2snr, 1./2.);
}



#endif /* MATCHGEN_HPP_ */
