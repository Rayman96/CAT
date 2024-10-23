// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.h"


using namespace std;
using namespace seal;

void example_ckks_basics()
{
    print_example_banner("Example: CKKS Basics");

    // 统计时间
    auto start = chrono::high_resolution_clock::now();
    /*
    In this example we demonstrate evaluating a polynomial function

        PI*x^3 + 0.4*x + 1

    on encrypted floating-point input data x for a set of 4096 equidistant points
    in the interval [0, 1]. This example demonstrates many of the main features
    of the CKKS scheme, but also the challenges in using it.

    We start by setting up the CKKS scheme.
    */
    EncryptionParameters parms(scheme_type::ckks);
    printf("init params end\n");


    /*
    We saw in `2_encoders.cpp' that multiplication in CKKS causes scales
    in ciphertexts to grow. The scale of any ciphertext must not get too close
    to the total size of coeff_modulus, or else the ciphertext simply runs out of
    room to store the scaled-up plaintext. The CKKS scheme provides a `rescale'
    functionality that can reduce the scale, and stabilize the scale expansion.

    Rescaling is a kind of modulus switch operation (recall `3_levels.cpp').
    As modulus switching, it removes the last of the primes from coeff_modulus,
    but as a side-effect it scales down the ciphertext by the removed prime.
    Usually we want to have perfect control over how the scales are changed,
    which is why for the CKKS scheme it is more common to use carefully selected
    primes for the coeff_modulus.

    More precisely, suppose that the scale in a CKKS ciphertext is S, and the
    last prime in the current coeff_modulus (for the ciphertext) is P. Rescaling
    to the next level changes the scale to S/P, and removes the prime P from the
    coeff_modulus, as usual in modulus switching. The number of primes limits
    how many rescalings can be done, and thus limits the multiplicative depth of
    the computation.

    It is possible to choose the initial scale freely. One good strategy can be
    to is to set the initial scale S and primes P_i in the coeff_modulus to be
    very close to each other. If ciphertexts have scale S before multiplication,
    they have scale S^2 after multiplication, and S^2/P_i after rescaling. If all
    P_i are close to S, then S^2/P_i is close to S again. This way we stabilize the
    scales to be close to S throughout the computation. Generally, for a circuit
    of depth D, we need to rescale D times, i.e., we need to be able to remove D
    primes from the coefficient modulus. Once we have only one prime left in the
    coeff_modulus, the remaining prime must be larger than S by a few bits to
    preserve the pre-decimal-point value of the plaintext.

    Therefore, a generally good strategy is to choose parameters for the CKKS
    scheme as follows:

        (1) Choose a 60-bit prime as the first prime in coeff_modulus. This will
            give the highest precision when decrypting;
        (2) Choose another 60-bit prime as the last element of coeff_modulus, as
            this will be used as the special prime and should be as large as the
            largest of the other primes;
        (3) Choose the intermediate primes to be close to each other.

    We use CoeffModulus::Create to generate primes of the appropriate size. Note
    that our coeff_modulus is 200 bits total, which is below the bound for our
    poly_modulus_degree: CoeffModulus::MaxBitCount(8192) returns 218.
    */
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);

    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40,  40, 60 }));

    /*
    We choose the initial scale to be 2^40. At the last level, this leaves us
    60-40=20 bits of precision before the decimal point, and enough (roughly
    10-20 bits) of precision after the decimal point. Since our intermediate
    primes are 40 bits (in fact, they are very close to 2^40), we can achieve
    scale stabilization as described above.
    */
    double scale = pow(2.0, 40);
    printf("set params end\n");
    SEALContext context(parms);
    print_parameters(context);
    cout << endl;
    Plaintext plain_coeff3;
    Plaintext x_plain;
    Plaintext y_plain;
    auto stage_1 = chrono::high_resolution_clock::now();
    auto duration_1 = chrono::duration_cast<chrono::microseconds>(stage_1 - start);
    cout << "stage_1 duration: " << duration_1.count() << " microseconds" << endl;


    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    auto stage_2 = chrono::high_resolution_clock::now();
    auto duration_2 = chrono::duration_cast<chrono::microseconds>(stage_2 - stage_1);
    cout << "stage_2 duration: " << duration_2.count() << " microseconds" << endl;

    PublicKey public_key;
    keygen.create_public_key(public_key);
    auto stage_3 = chrono::high_resolution_clock::now();
    auto duration_3 = chrono::duration_cast<chrono::microseconds>(stage_3 - stage_2);
    cout << "stage_3 duration: " << duration_3.count() << " microseconds" << endl;

    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    auto stage_4 = chrono::high_resolution_clock::now();
    auto duration_4 = chrono::duration_cast<chrono::microseconds>(stage_4 - stage_3);
    cout << "stage_4 duration: " << duration_4.count() << " microseconds" << endl;


    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    auto stage_5 = chrono::high_resolution_clock::now();
    auto duration_5 = chrono::duration_cast<chrono::microseconds>(stage_5 - stage_4);
    cout << "stage_5 duration: " << duration_5.count() << " microseconds" << endl;


    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;
    auto stage_6 = chrono::high_resolution_clock::now();
    auto duration_6 = chrono::duration_cast<chrono::microseconds>(stage_6 - stage_5);
    cout << "stage_6 duration: " << duration_6.count() << " microseconds" << endl;


    vector<double> input;
    vector<double> input2;
    vector<double> input_01;
    input.reserve(slot_count);
    input2.reserve(slot_count);
    input_01.reserve(slot_count);
    auto stage_7 = chrono::high_resolution_clock::now();
    auto duration_7 = chrono::duration_cast<chrono::microseconds>(stage_7 - stage_6);
    cout << "stage_7 duration: " << duration_7.count() << " microseconds" << endl;


    

    double curr_point = 0;
    double step_size = 1.0 / (static_cast<double>(slot_count) - 1);
    for (size_t i = 0; i < slot_count; i++)
    {
        input.push_back(curr_point*2);
        input2.push_back(curr_point);
        input_01.push_back(0.1);
        
        curr_point += step_size;
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    encoder.encode(input, scale, x_plain);
    encoder.encode(input2, scale, y_plain);
    // encoder.encode(input_01, scale, plain_coeff3);
    encoder.encode(0.1, scale, plain_coeff3);

    Plaintext plain_coeff1;
    encoder.encode(1, scale, plain_coeff1);

    Ciphertext x_encrypted;
    Ciphertext y_encrypted;
    encryptor.encrypt(x_plain, x_encrypted);
    encryptor.encrypt(y_plain, y_encrypted);

    // Ciphertext z_encrypted;
    // evaluator.sub(x_encrypted, y_encrypted, z_encrypted);

    // cout << z_encrypted.size() << endl;
    // evaluator.relinearize_inplace(z_encrypted, relin_keys);

    // Ciphertext z_suqare;
    // evaluator.square(z_encrypted, z_suqare);
    // evaluator.rescale_to_next_inplace(z_suqare);
    // evaluator.relinearize_inplace(z_suqare, relin_keys);

//   eq_list.emplace_back( (-1.0)/36.0 * (input_expr_z + 3.0) * (input_expr_z + 2.0) * (input_expr_z + 1.0) * (input_expr_z - 1.0) * (input_expr_z - 2.0) * (input_expr_z - 3.0));

//   lt_list.emplace_back((1.0/72.0 * input_expr_z * input_expr_z + 3.0/40.0 * input_expr_z + 37.0/360.0) * input_expr_z * (input_expr_z - 1.0) *(input_expr_z - 2.0) * (input_expr_z - 3.0));

    Ciphertext z_rotate;
    Ciphertext z_rotate2;
    Ciphertext z_squared;
    Ciphertext x_01;
    Ciphertext x_02;
    Ciphertext x_03;

    z_rotate = x_encrypted;

    cudaStream_t stream1;
    cudaStream_t stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    evaluator.multiply_inplace(x_encrypted, y_encrypted, stream1);

    evaluator.multiply_inplace(z_rotate, y_encrypted, stream2);


    // evaluator.rotate_vector_inplace(z_rotate, 1, gal_keys);
    // evaluator.mod_switch_to_next_inplace(z_rotate);

    // Ciphertext x_mod_switch;
    // evaluator.mod_switch_to_next(x_encrypted, x_mod_switch);

    // evaluator.square(x_encrypted, z_squared);

    // evaluator.add_plain(x_encrypted, y_plain, x_01);

    // evaluator.relinearize_inplace(x_01, relin_keys);
    // evaluator.rescale_to_next_inplace(x_01);
    // evaluator.rescale_to(x_encrypted, x_01.parms_id(), x_mod_switch);


    // evaluator.add(x_01, x_mod_switch, x_03);


    // evaluator.rescale_to_next_inplace(x_01);

    // evaluator.multiply_plain(x_encrypted, plain_coeff1, x_02);

    // evaluator.mod_switch_to_next(x_encrypted, x_mod_switch);

    // evaluator.multiply_inplace(x_01, x_mod_switch);

    // evaluator.mod_switch_to_next_inplace(x_02);



    // evaluator.mod_switch_to_next_inplace(x_encrypted);

    // evaluator.multiply_plain_inplace(x_encrypted, plain_coeff3);
    // evaluator.rescale_to_next_inplace(x_encrypted);
    // // evaluator.relinearize_inplace(x_encrypted, relin_keys);

    // evaluator.rescale_to_next_inplace(x_01);
    // evaluator.relinearize_inplace(x_01, relin_keys);




    // evaluator.mod_switch_to_inplace(x_encrypted, x_01.parms_id());
    // cout << x_encrypted.scale() << "," << x_01.scale() << endl;
    // evaluator.rescale_to_inplace(x_encrypted, x_01.parms_id());
    // evaluator.add_inplace(x_01, x_encrypted);


    // evaluator.rotate_vector(x_encrypted, 2, gal_keys,z_rotate);
    // evaluator.rotate_vector(x_encrypted, 3, gal_keys,z_rotate2);


    // cout << "finish rotate" << endl;
    // Ciphertext z_tmp;
    // evaluator.multiply(y_encrypted, z_rotate, z_tmp);
    // evaluator.rescale_to_next_inplace(z_tmp);
    // evaluator.relinearize_inplace(z_tmp, relin_keys);

    // Ciphertext z_tmp2;
    // evaluator.mod_switch_to_next_inplace(x_encrypted);
    // evaluator.rescale_to_next_inplace(x_encrypted);
    // evaluator.add(z_tmp, x_encrypted, z_tmp2);


    // evaluator.mod_switch_to_next_inplace(z_rotate2);
    // evaluator.multiply(z_tmp, z_rotate2, z_tmp2);

    Plaintext plain_tmp_result;
    decryptor.decrypt(z_rotate, plain_tmp_result);
    cout << "decrypt---------------" << endl;
    auto decrypt = chrono::high_resolution_clock::now();

    vector<double> tmp_result;

    encoder.decode(plain_tmp_result, tmp_result);
    print_vector(tmp_result, 3, 7);


    // cout << "Evaluating polynomial PI*x^3 + 0.4x + 1 ..." << endl;

    // /*
    // We create plaintexts for PI, 0.4, and 1 using an overload of CKKSEncoder::encode
    // that encodes the given floating-point value to every slot in the vector.
    // */
    // // Plaintext plain_coeff3, plain_coeff1, plain_coeff0;



    // // encoder.encode(3.14159265, scale, plain_coeff3);
    // // encoder.encode(0.4, scale, plain_coeff1);
    // cout << "--------------------------finish init params---------------------------------" << endl;
    // auto stage_encode1_start = chrono::high_resolution_clock::now();

    // encoder.encode(1.0, scale, plain_coeff0);
    // auto stage_encode1_end = chrono::high_resolution_clock::now();
    // auto duration_encode_1 = chrono::duration_cast<chrono::microseconds>(stage_encode1_end - stage_encode1_start);
    // cout << "encode_1 duration: " << duration_encode_1.count() << " microseconds" << endl;

    // // vector<double> result;
    // // encoder.decode(plain_coeff3, result);
    
    // // for (double item : result) {
    // //     cout << "decode result: " << item << endl;
    // // }

    // // print_line(__LINE__);
    // // cout << "Encode input vectors." << endl;
    // encoder.encode(input, scale, x_plain);
    // auto stage_8 = chrono::high_resolution_clock::now();
    // auto duration_8 = chrono::duration_cast<chrono::microseconds>(stage_8 - stage_encode1_end);
    // cout << "duration_encode_2 duration: " << duration_8.count() << " microseconds" << endl;


    // Ciphertext x1_encrypted;
    // encryptor.encrypt(x_plain, x1_encrypted);
    // auto stage_9 = chrono::high_resolution_clock::now();
    // auto duration_9 = chrono::duration_cast<chrono::microseconds>(stage_9 - stage_8);
    // cout << "stage_9 duration: " << duration_9.count() << " microseconds" << endl;
    // // x1_encrypted.to_gpu();



    // // Plaintext plain_tmp_result;
    // // decryptor.decrypt(x1_encrypted, plain_tmp_result);
    // // cout << "decrypt---------------" << endl;
    // // auto decrypt = chrono::high_resolution_clock::now();
    // // auto duration_decrypt = chrono::duration_cast<chrono::microseconds>(decrypt - stage_9);
    // // cout << "duration_decrypt duration: " << duration_decrypt.count() << " microseconds" << endl;

    // // vector<double> tmp_result;
    // // encoder.decode(plain_tmp_result, tmp_result);
    // // auto stage_decode = chrono::high_resolution_clock::now();
    // // auto duration_decode = chrono::duration_cast<chrono::microseconds>(stage_decode - decrypt);
    // // cout << "duration_decode duration: " << duration_decode.count() << " microseconds" << endl;
    // // cout << "    + Computed result ...... Correct." << endl;
    // // print_vector(tmp_result, 3, 7);



    // /*
    // To compute x^3 we first compute x^2 and relinearize. However, the scale has
    // now grown to 2^80.
    // */
    // // Ciphertext x3_encrypted;
    // // print_line(__LINE__);
    // // cout << "Compute x^2 and relinearize:" << endl;
    // // evaluator.square(x1_encrypted, x3_encrypted);
    // // evaluator.relinearize_inplace(x3_encrypted, relin_keys);
    // // cout << "    + Scale of x^2 before rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;
    // // auto stage_10 = chrono::high_resolution_clock::now();
    // // auto duration_10 = chrono::duration_cast<chrono::microseconds>(stage_10 - stage_9);
    // // cout << "stage_10 duration: " << duration_10.count() << " microseconds" << endl;




    // // /*
    // // Now rescale; in addition to a modulus switch, the scale is reduced down by
    // // a factor equal to the prime that was switched away (40-bit prime). Hence, the
    // // new scale should be close to 2^40. Note, however, that the scale is not equal
    // // to 2^40: this is because the 40-bit prime is only close to 2^40.
    // // */
    // // print_line(__LINE__);
    // // cout << "Rescale x^2." << endl;
    // // evaluator.rescale_to_next_inplace(x3_encrypted);
    // // cout << "    + Scale of x^2 after rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;
    // // auto stage_11 = chrono::high_resolution_clock::now();
    // // auto duration_11 = chrono::duration_cast<chrono::microseconds>(stage_11 - stage_10);
    // // cout << "stage_11 duration: " << duration_11.count() << " microseconds" << endl;




    // // /*
    // // Now x3_encrypted is at a different level than x1_encrypted, which prevents us
    // // from multiplying them to compute x^3. We could simply switch x1_encrypted to
    // // the next parameters in the modulus switching chain. However, since we still
    // // need to multiply the x^3 term with PI (plain_coeff3), we instead compute PI*x
    // // first and multiply that with x^2 to obtain PI*x^3. To this end, we compute
    // // PI*x and rescale it back from scale 2^80 to something close to 2^40.
    // // */
    // // print_line(__LINE__);
    // // cout << "Compute and rescale PI*x." << endl;
    // // Ciphertext x1_encrypted_coeff3;
    // // evaluator.multiply_plain(x1_encrypted, plain_coeff3, x1_encrypted_coeff3);
    // // cout << "    + Scale of PI*x before rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << endl;
    // // // auto stage_12 = chrono::high_resolution_clock::now();
    // // // auto duration_12 = chrono::duration_cast<chrono::microseconds>(stage_12 - stage_11);
    // // // cout << "stage_12 duration: " << duration_12.count() << " microseconds" << endl;



    // // print_line(__LINE__);
    // // cout << "Decrypt and decode PI*x." << endl;
    // // cout << "    + Expected result:" << endl;
    // // vector<double> true_tmp_result;
    // // for (size_t i = 0; i < input.size(); i++)
    // // {
    // //     double x = input[i];
    // //     // true_tmp_result.push_back(3.14159265 * x);
    // //     true_tmp_result.push_back(x + 1);
    // //     // true_tmp_result.push_back(x * x);
    // // }
    // // print_vector(true_tmp_result, 3, 7);


    // /*
    // Decrypt, decode, and print the result.
    // */





    // // evaluator.rescale_to_next_inplace(x1_encrypted_coeff3);
    // // cout << "    + Scale of PI*x after rescale: " << log2(x1_encrypted_coeff3.scale()) << " bits" << endl;
    // // auto stage_13 = chrono::high_resolution_clock::now();
    // // auto duration_13 = chrono::duration_cast<chrono::microseconds>(stage_13 - stage_12);
    // // cout << "stage_13 duration: " << duration_13.count() << " microseconds" << endl;



    // // /*
    // // Since x3_encrypted and x1_encrypted_coeff3 have the same exact scale and use
    // // the same encryption parameters, we can multiply them together. We write the
    // // result to x3_encrypted, relinearize, and rescale. Note that again the scale
    // // is something close to 2^40, but not exactly 2^40 due to yet another scaling
    // // by a prime. We are down to the last level in the modulus switching chain.
    // // */
    // // print_line(__LINE__);
    // // cout << "Compute, relinearize, and rescale (PI*x)*x^2." << endl;
    // // evaluator.multiply_inplace(x3_encrypted, x1_encrypted_coeff3);
    // // auto stage_14 = chrono::high_resolution_clock::now();
    // // auto duration_14 = chrono::duration_cast<chrono::microseconds>(stage_14 - stage_13);
    // // cout << "stage_14 duration: " << duration_14.count() << " microseconds" << endl;


    // // evaluator.relinearize_inplace(x3_encrypted, relin_keys);
    // // cout << "    + Scale of PI*x^3 before rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;
    // // auto stage_15 = chrono::high_resolution_clock::now();
    // // auto duration_15 = chrono::duration_cast<chrono::microseconds>(stage_15 - stage_14);
    // // cout << "stage_15 duration: " << duration_15.count() << " microseconds" << endl;



    // // evaluator.rescale_to_next_inplace(x3_encrypted);
    // // cout << "    + Scale of PI*x^3 after rescale: " << log2(x3_encrypted.scale()) << " bits" << endl;
    // // auto stage_16 = chrono::high_resolution_clock::now();
    // // auto duration_16 = chrono::duration_cast<chrono::microseconds>(stage_16 - stage_15);
    // // cout << "stage_16 duration: " << duration_16.count() << " microseconds" << endl;



    // // /*
    // // Next we compute the degree one term. All this requires is one multiply_plain
    // // with plain_coeff1. We overwrite x1_encrypted with the result.
    // // */
    // // print_line(__LINE__);
    // // cout << "Compute and rescale 0.4*x." << endl;
    // // evaluator.multiply_plain_inplace(x1_encrypted, plain_coeff1);
    // // auto stage_17 = chrono::high_resolution_clock::now();
    // // auto duration_17 = chrono::duration_cast<chrono::microseconds>(stage_17 - stage_16);
    // // cout << "stage_17 duration: " << duration_17.count() << " microseconds" << endl;



    // // cout << "    + Scale of 0.4*x before rescale: " << log2(x1_encrypted.scale()) << " bits" << endl;
    // // evaluator.rescale_to_next_inplace(x1_encrypted);
    // // auto stage_18 = chrono::high_resolution_clock::now();
    // // auto duration_18 = chrono::duration_cast<chrono::microseconds>(stage_18 - stage_17);
    // // cout << "stage_18 duration: " << duration_18.count() << " microseconds" << endl;


    // // cout << "    + Scale of 0.4*x after rescale: " << log2(x1_encrypted.scale()) << " bits" << endl;

    // /*
    // Now we would hope to compute the sum of all three terms. However, there is
    // a serious problem: the encryption parameters used by all three terms are
    // different due to modulus switching from rescaling.

    // Encrypted addition and subtraction require that the scales of the inputs are
    // the same, and also that the encryption parameters (parms_id) match. If there
    // is a mismatch, Evaluator will throw an exception.
    // */
    // // cout << endl;
    // // print_line(__LINE__);
    // // cout << "Parameters used by all three terms are different." << endl;
    // // cout << "    + Modulus chain index for x3_encrypted: "
    // //      << context.get_context_data(x3_encrypted.parms_id())->chain_index() << endl;
    // // cout << "    + Modulus chain index for x1_encrypted: "
    // //      << context.get_context_data(x1_encrypted.parms_id())->chain_index() << endl;
    // // cout << "    + Modulus chain index for plain_coeff0: "
    // //      << context.get_context_data(plain_coeff0.parms_id())->chain_index() << endl;
    // // cout << endl;

    // // auto stage_19 = chrono::high_resolution_clock::now();
    // // auto duration_19 = chrono::duration_cast<chrono::microseconds>(stage_19 - stage_18);
    // // cout << "stage_19 duration: " << duration_19.count() << " microseconds" << endl;



    // /*
    // Let us carefully consider what the scales are at this point. We denote the
    // primes in coeff_modulus as P_0, P_1, P_2, P_3, in this order. P_3 is used as
    // the special modulus and is not involved in rescalings. After the computations
    // above the scales in ciphertexts are:

    //     - Product x^2 has scale 2^80 and is at level 2;
    //     - Product PI*x has scale 2^80 and is at level 2;
    //     - We rescaled both down to scale 2^80/P_2 and level 1;
    //     - Product PI*x^3 has scale (2^80/P_2)^2;
    //     - We rescaled it down to scale (2^80/P_2)^2/P_1 and level 0;
    //     - Product 0.4*x has scale 2^80;
    //     - We rescaled it down to scale 2^80/P_2 and level 1;
    //     - The contant term 1 has scale 2^40 and is at level 2.

    // Although the scales of all three terms are approximately 2^40, their exact
    // values are different, hence they cannot be added together.
    // */
    // // print_line(__LINE__);
    // // cout << "The exact scales of all three terms are different:" << endl;
    // // ios old_fmt(nullptr);
    // // old_fmt.copyfmt(cout);
    // // cout << fixed << setprecision(10);
    // // cout << "    + Exact scale in PI*x^3: " << x3_encrypted.scale() << endl;
    // // cout << "    + Exact scale in  0.4*x: " << x1_encrypted.scale() << endl;
    // // cout << "    + Exact scale in      1: " << plain_coeff0.scale() << endl;
    // // cout << endl;
    // // cout.copyfmt(old_fmt);

    // // auto stage_20 = chrono::high_resolution_clock::now();
    // // auto duration_20 = chrono::duration_cast<chrono::microseconds>(stage_20 - stage_19);
    // // cout << "stage_20 duration: " << duration_20.count() << " microseconds" << endl;


    // /*
    // There are many ways to fix this problem. Since P_2 and P_1 are really close
    // to 2^40, we can simply "lie" to Microsoft SEAL and set the scales to be the
    // same. For example, changing the scale of PI*x^3 to 2^40 simply means that we
    // scale the value of PI*x^3 by 2^120/(P_2^2*P_1), which is very close to 1.
    // This should not result in any noticeable error.

    // Another option would be to encode 1 with scale 2^80/P_2, do a multiply_plain
    // with 0.4*x, and finally rescale. In this case we would need to additionally
    // make sure to encode 1 with appropriate encryption parameters (parms_id).

    // In this example we will use the first (simplest) approach and simply change
    // the scale of PI*x^3 and 0.4*x to 2^40.
    // */
    // // print_line(__LINE__);
    // // cout << "Normalize scales to 2^40." << endl;
    // // x3_encrypted.scale() = pow(2.0, 40);
    // // x1_encrypted.scale() = pow(2.0, 40);

    // // auto stage_21 = chrono::high_resolution_clock::now();
    // // auto duration_21 = chrono::duration_cast<chrono::microseconds>(stage_21 - stage_20);
    // // cout << "stage_21 duration: " << duration_21.count() << " microseconds" << endl;


    // /*
    // We still have a problem with mismatching encryption parameters. This is easy
    // to fix by using traditional modulus switching (no rescaling). CKKS supports
    // modulus switching just like the BFV scheme, allowing us to switch away parts
    // of the coefficient modulus when it is simply not needed.
    // */
    // // print_line(__LINE__);
    // // cout << "Normalize encryption parameters to the lowest level." << endl;

    // auto start_mod_switch = chrono::high_resolution_clock::now();

    // parms_id_type last_parms_id = x1_encrypted.parms_id();
    // // evaluator.mod_switch_to_inplace(x1_encrypted, last_parms_id);
    // evaluator.mod_switch_to_inplace(plain_coeff0, last_parms_id);
    // auto stage_22 = chrono::high_resolution_clock::now();
    // auto duration_22 = chrono::duration_cast<chrono::microseconds>(stage_22 - start_mod_switch);
    // cout << "stage_22 duration: " << duration_22.count() << " microseconds" << endl;

    // // /*
    // // All three ciphertexts are now compatible and can be added.
    // // */
    // // print_line(__LINE__);
    // // cout << "Compute PI*x^3 + 0.4*x + 1." << endl;
    // // Ciphertext encrypted_result;
    // // evaluator.add(x3_encrypted, x1_encrypted, encrypted_result);
    // // evaluator.add_plain_inplace(x1_encrypted, plain_coeff0);
    // evaluator.sub_plain_inplace(x1_encrypted, plain_coeff0);

    // auto stage_23 = chrono::high_resolution_clock::now();
    // auto duration_23 = chrono::duration_cast<chrono::microseconds>(stage_23 - stage_22);
    // cout << "stage_23 duration: " << duration_23.count() << " microseconds" << endl;

    // evaluator.negate_inplace(x1_encrypted);
    // auto stage_24 = chrono::high_resolution_clock::now();
    // auto duration_24 = chrono::duration_cast<chrono::microseconds>(stage_24 - stage_23);
    // cout << "stage_24 duration: " << duration_24.count() << " microseconds" << endl;


    // // evaluator.multiply_plain_inplace(x1_encrypted, plain_coeff0);
    // // auto stage_25 = chrono::high_resolution_clock::now();
    // // auto duration_25 = chrono::duration_cast<chrono::microseconds>(stage_25 - stage_24);
    // // cout << "stage_25 duration: " << duration_25.count() << " microseconds" << endl;

    // // evaluator.rescale_to_next_inplace(x1_encrypted);
    // // auto stage_rescale = chrono::high_resolution_clock::now();
    // // auto duration_rescale = chrono::duration_cast<chrono::microseconds>(stage_rescale - stage_25);
    // // cout << "duration_rescale duration: " << duration_rescale.count() << " microseconds" << endl;

    // evaluator.square_inplace(x1_encrypted);
    // auto stage_26 = chrono::high_resolution_clock::now();
    // auto duration_26 = chrono::duration_cast<chrono::microseconds>(stage_26 - stage_24);
    // cout << "stage_26 duration: " << duration_26.count() << " microseconds" << endl;



    // cout << "decrypt---------------" << endl;
    // Plaintext plain_tmp_result;
    // decryptor.decrypt(x1_encrypted, plain_tmp_result);
    // auto decrypt = chrono::high_resolution_clock::now();
    // auto duration_decrypt = chrono::duration_cast<chrono::microseconds>(decrypt - stage_26);
    // cout << "duration_decrypt duration: " << duration_decrypt.count() << " microseconds" << endl;

    // vector<double> tmp_result;
    // encoder.decode(plain_tmp_result, tmp_result);
    // auto stage_decode = chrono::high_resolution_clock::now();
    // auto duration_decode = chrono::duration_cast<chrono::microseconds>(stage_decode - decrypt);
    // cout << "duration_decode duration: " << duration_decode.count() << " microseconds" << endl;
    // cout << "    + Computed result ...... Correct." << endl;
    // print_vector(tmp_result, 3, 7);

    // /*
    // First print the true result.
    // */
    // // Plaintext plain_result;
    // // print_line(__LINE__);
    // // cout << "Decrypt and decode PI*x^3 + 0.4x + 1." << endl;
    // // cout << "    + Expected result:" << endl;
    // // vector<double> true_result;
    // // for (size_t i = 0; i < input.size(); i++)
    // // {
    // //     double x = input[i];
    // //     true_result.push_back((3.14159265 * x * x + 0.4) * x + 1);
    // // }
    // // print_vector(true_result, 3, 7);

    // // auto stage_24 = chrono::high_resolution_clock::now();
    // // auto duration_24 = chrono::duration_cast<chrono::microseconds>(stage_24 - stage_23);
    // // cout << "stage_24 duration: " << duration_24.count() << " microseconds" << endl;


    // /*
    // Decrypt, decode, and print the result.
    // // */
    // // decryptor.decrypt(encrypted_result, plain_result);
    // // vector<double> result;
    // // encoder.decode(plain_result, result);
    // // cout << "    + Computed result ...... Correct." << endl;
    // // print_vector(result, 3, 7);

    // // auto end = chrono::high_resolution_clock::now();
    // // auto duration = chrono::duration_cast<chrono::microseconds>(end - stage_24);
    // // cout << "stage_25 duration: " << duration.count() << " microseconds" << endl;

    // /*
    // While we did not show any computations on complex numbers in these examples,
    // the CKKSEncoder would allow us to have done that just as easily. Additions
    // and multiplications of complex numbers behave just as one would expect.
    // */
}

int main() {
    example_ckks_basics();
}