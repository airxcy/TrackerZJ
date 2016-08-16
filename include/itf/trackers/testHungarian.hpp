#ifndef TESTHUNGARIAN
#define TESTHUNGARIAN
/*
 *   Copyright (c) 2007 John Weaver
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

/*
 * Some example code.
 *
 */

#include <iostream>
#include <cstdlib>
//#include <random>
#include <opencv2/opencv.hpp>


//boost 
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>
#include <boost/random/binomial_distribution.hpp>

//#include "HungarianAlign.hpp"


boost::random::mt19937 eng;
boost::random::uniform_int_distribution<> dist(1,50);

void testHungarian(int nrows, int ncols)
{
	
	std::vector< std::vector<double> > matrix(nrows,std::vector<double>(ncols));
    
    //  srandom(time(NULL)); // Seed random number generator.


    // Initialize matrix with random values.
    for ( int row = 0 ; row < nrows ; row++ ) {
        for ( int col = 0 ; col < ncols ; col++ ) {
            matrix[row][col] = (float)dist(eng)/15;
        }
    }
    
    // Display begin matrix state.
    for ( int row = 0 ; row < nrows ; row++ ) {
        for ( int col = 0 ; col < ncols ; col++ ) {
            std::cout.width(2);
            std::cout << (float)matrix[row][col] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Apply Munkres algorithm to matrix.
	HungarianAlign HA;
	
	std::vector<int> Assignment;
		
	cout << HA.Solve(matrix,Assignment,HungarianAlign::optimal) << endl;

    
    // Display solved matrix.
    for ( int row = 0 ; row < nrows ; row++ ) {
        for ( int col = 0 ; col < ncols ; col++ ) {
            std::cout.width(2);
            std::cout << matrix[row][col] << ",";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    
    
    for ( int row = 0 ; row < nrows ; row++ ) {
        int rowcount = 0;
        for ( int col = 0 ; col < ncols ; col++  ) {
            if ( matrix[row][col] == 0 )
                rowcount++;
        }
        if ( rowcount != 1 )
            std::cerr << "Row " << row << " has " << rowcount << " columns that have been matched." << std::endl;
    }
    
    for ( int col = 0 ; col < ncols ; col++ ) {
        int colcount = 0;
        for ( int row = 0 ; row < nrows ; row++ ) {
            if ( matrix[row][col] == 0 )
                colcount++;
        }
        if ( colcount != 1 )
            std::cerr << "Column " << col << " has " << colcount << " rows that have been matched." << std::endl;
    }

	//OR
			
	// Output the result
	for(int x=0; x<nrows; x++)
	{
		std::cout << x << ":" << Assignment[x] << "\t";
	}
}

#endif
