using System;
using System.Data;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using Supercluster.KDTree;

namespace johns_icp
{
    class JohnsICP {
        static MatrixBuilder<double> matrixBuilder;
        static VectorBuilder<double> vectorBuilder;
        static RandomSource randomSource;

        static
            JohnsICP()
        {
            matrixBuilder = Matrix<double>.Build;
            vectorBuilder = Vector<double>.Build;
            randomSource = new Mcg31m1();
        }

        public static Matrix<double> 
            ClosestPointMatrix(
                Matrix<double> M_points,  // will find the point in here...
                Matrix<double> P_points   // ...that is closest to each point in here
            )
        {
            int dimension = P_points.RowCount;
            var arrayOfArraysP = P_points.ToColumnArrays();
            var arrayOfArraysM = M_points.ToColumnArrays();
            var arrayOfStringsM = arrayOfArraysM.Select(p => p.ToString()).ToArray();

            Func<double[], double[], double> 
                EuclideanDistanceSquared = (x, y) =>
            {
                double dist = 0;
                for (int i = 0; i < dimension; i++) { dist += (x[i] - y[i]) * (x[i] - y[i]); }
                return dist;
            };

            var tree = 
                new KDTree<double, string>(
                    dimension, 
                    arrayOfArraysM, 
                    arrayOfStringsM, 
                    EuclideanDistanceSquared
                );

            var Y = matrixBuilder.Dense(dimension, arrayOfArraysP.Length);
            for (int j = 0; j < arrayOfArraysP.Length; j++)
            {
                Tuple<double[], string> closest = tree.NearestNeighbors(arrayOfArraysP[j], 1)[0];
                Y.SetColumn(j, closest.Item1);
            }            
            return Y;
        }

        public static Vector<double> 
            Centroid(
                Matrix<double> matrix
            )
        {
            return matrix.RowSums().Divide(matrix.ColumnCount);
        }

        public static Matrix<double>
            AddVectorToColumns(
                Matrix<double> matrix,
                Vector<double> vector
            )
        {
            Matrix<double> m = matrixBuilder.Dense(matrix.RowCount, matrix.ColumnCount);
            for (int j = 0; j < matrix.ColumnCount; j++)
            {
                m.SetColumn(j, matrix.Column(j) + vector);
            }
            return m;
        }

        public static Matrix<double>
            NFromS(
                // see slide 84 of http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_ICP09.pdf
                // or page 8 of https://www.researchgate.net/publication/230600780_Closed-Form_Solution_of_Absolute_Orientation_Using_Unit_Quaternions
                Matrix<double> S
            )
        {
            double Sxx = 0, Sxy = 0, Sxz = 0;
            double Syx = 0, Syy = 0, Syz = 0;
            double Szx = 0, Szy = 0, Szz = 0;

            int dimension = S.RowCount;

            if (dimension == 2) {
                Sxx = S[0, 0]; Sxy = S[0, 1];
                Syx = S[1, 0]; Syy = S[1, 1];
            }

            else {
                Sxx = S[0, 0]; Sxy = S[0, 1]; Sxz = S[0, 2];
                Syx = S[1, 0]; Syy = S[1, 1]; Syz = S[1, 2];
                Szx = S[2, 0]; Szy = S[2, 1]; Szz = S[2, 2];
            }

            return matrixBuilder.DenseOfArray(
                new double[,] {
                        { Sxx + Syy + Szz,  Syz - Szy,       -Sxz + Szx,        Sxy - Syx},
                        { Syz - Szy,        Sxx - Syy - Szz,  Sxy + Syx,        Sxz + Szx},
                        {-Sxz + Szx,        Sxy + Syx,       -Sxx + Syy - Szz,  Syz + Szy},
                        { Sxy - Syx,        Sxz + Szx,        Syz + Szy,       -Sxx - Syy + Szz} 
                }
            );
        }

        public static Vector<double>
            LargestEigenvectorOfSymmetricMatrix(
                Matrix<double> matrix
            )
        {
            return matrix.Evd().EigenVectors.Column(matrix.ColumnCount - 1);
        }

        public static Tuple<Matrix<double>, Matrix<double>>
            QAndQbarFromQuaternion(
                // see slides 38 and 46 of http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_ICP09.pdf
                // or pages 7 & 8 of https://www.researchgate.net/publication/230600780_Closed-Form_Solution_of_Absolute_Orientation_Using_Unit_Quaternions
                Vector<double> q
            )
        {
            double q0 = q[0]; double q1 = q[1]; double q2 = q[2]; double q3 = q[3];
            Matrix<double> Q = matrixBuilder.DenseOfArray(
                new double[,] {
                    { q0, -q1, -q2, -q3 },
                    { q1,  q0, -q3,  q2 },
                    { q2,  q3,  q0, -q1 },
                    { q3, -q2,  q1,  q0 }
                }
            );
            Matrix<double> Qbar = matrixBuilder.DenseOfArray(
                new double[,] { 
                    { q0, -q1, -q2, -q3 },
                    { q1,  q0,  q3, -q2 },
                    { q2, -q3,  q0,  q1 },
                    { q3,  q2, -q1,  q0 }
                }
            );
            return new Tuple<Matrix<double>, Matrix<double>>(Q, Qbar);
        }

        public static Matrix<double>
            RotationMatrixFromQuaternion(Vector<double> q)
        {
            Tuple<Matrix<double>, Matrix<double>> Q_and_Qbar = QAndQbarFromQuaternion(q);
            Matrix<double> Q = Q_and_Qbar.Item1;
            Matrix<double> Qbar = Q_and_Qbar.Item2;
            return (Qbar.Transpose() * Q).RemoveColumn(0).RemoveRow(0);
        }

        public static Double
            SumOfColumnNorms(Matrix<double> matrix)
        {
            double sum = 0;
            for (int j = 0; j < matrix.ColumnCount; j++) {
                sum += matrix.Column(j).Norm(2);
            }
            return sum;
        }

        public static Vector<double>
            QuaternionConstrainedToY(Vector<double> q)
        {
            Vector<double> w = vectorBuilder.DenseOfArray(new double[] { q[0], 0, q[2], 0 });
            // maybe-not-necessary-hack to avoid division by 0:
            if (w[0] * w[0] + w[2] * w[2] < 0.01) {
                w[2] = 0.1;
            }
            return w / w.Norm(2);
        }

        public static Tuple<Matrix<double>, Vector<double>, double> 
            NextRotationAndTranslationAndError(
                Matrix<double> M_points, 
                Matrix<double> P_points,
                bool constrain_rotations_to_Y_axis=false
            )
        {
            int dimension = P_points.RowCount;
            Matrix<double> Y = ClosestPointMatrix(M_points, P_points);
            Vector<double> P_centroid = Centroid(P_points);
            Vector<double> Y_centroid = Centroid(Y);
            Matrix<double> P_centered = AddVectorToColumns(P_points, -P_centroid);
            Matrix<double> Y_centered = AddVectorToColumns(Y, -Y_centroid);
            Matrix<double> S = P_centered * Y_centered.Transpose();
            Matrix<double> N = NFromS(S);
            Vector<double> q = LargestEigenvectorOfSymmetricMatrix(N);
            if (constrain_rotations_to_Y_axis && dimension == 3) {
                q = QuaternionConstrainedToY(q);
            }
            Matrix<double> R = RotationMatrixFromQuaternion(q);
            if (dimension == 2) {
                R = R.SubMatrix(0, 2, 0, 2);
            }
            Vector<double> t = Y_centroid - R * P_centroid;
            double err = SumOfColumnNorms(Y_centered - R * P_centered);
            // note: the above is equal to: SumOfColumnNorms(Y - (R * P_points + t))
            // ...where the "+ t" at the end must be computed using AddVectorToColumns
            return new Tuple<Matrix<double>, Vector<double>, double>(R, t, err);
        }

        static Tuple<Matrix<double>, Vector<double>, double, int>
            RunJohnICP(
                Matrix<double> M_points, 
                Matrix<double> P_points,
                int max_iterations,
                double error_threshold_per_point,
                double lack_of_progress_threshold,
                bool constrain_rotations_to_Y_axis=false
            )
        {
            int dimension = P_points.RowCount;
            Matrix<double> cumulative_R = matrixBuilder.DenseIdentity(dimension);
            Vector<double> cumulative_t = Centroid(M_points) - Centroid(P_points);
            double latest_err = Double.PositiveInfinity;
            double error_five_ago = -1;
            int iteration_number = 0;
            while (true) {
                iteration_number++;
                Tuple<Matrix<double>, Vector<double>, double> triple = NextRotationAndTranslationAndError(
                    M_points, 
                    AddVectorToColumns(cumulative_R * P_points, cumulative_t), 
                    constrain_rotations_to_Y_axis
                );
                Matrix<double> latest_R = triple.Item1;
                Vector<double> latest_t = triple.Item2;
                latest_err = triple.Item3;
                cumulative_R = latest_R * cumulative_R;
                cumulative_t = latest_R * cumulative_t + latest_t;
                if (latest_err / P_points.ColumnCount < error_threshold_per_point) {
                    break;
                }
                if (iteration_number % 5 == 1) {
                    if (latest_err / error_five_ago > lack_of_progress_threshold) {
                        break;
                    }
                    error_five_ago = latest_err;
                }
                if (iteration_number >= max_iterations) {
                    break;
                }
            }
            return new Tuple<Matrix<double>, Vector<double>, double, int>(
                cumulative_R, 
                cumulative_t, 
                latest_err / P_points.ColumnCount, 
                iteration_number
            );
        }

        static double
            RandomInInterval(
                double min,
                double max
            )
        {
            return min + (max - min) * randomSource.NextDouble();
        }

        static Vector<double>
            RandomVectorInInterval(
                int dimension,
                double min,
                double max
            )
        {
            Vector<double> v = vectorBuilder.Dense(dimension);
            for (int i = 0; i < dimension; i++)
            {
                v[i] = RandomInInterval(min, max);
            }
            return v;
        }

        static Matrix<double>
            RandomMatrixInInterval(
                int num_rows,
                int num_columns,
                double min,
                double max
            )
        {
            Matrix<double> m = matrixBuilder.Dense(num_rows, num_columns);
            for (int j = 0; j < num_columns; j++) {
                m.SetColumn(j, RandomVectorInInterval(num_rows, min, max));
            }
            return m;
        }

        static Matrix<double>
            MatrixPlusRandomMatrixInInterval(
                Matrix<double> matrix,
                double min,
                double max
            )
        {
            return matrix + RandomMatrixInInterval(matrix.RowCount, matrix.ColumnCount, min, max);
        }

        static Vector<double>
            RandomQuaternion(
                double q0_truncation=0
            )
        {
            double q0 = 0, q1 = 0, q2 = 0, q3 = 0;
            do {
                q0 = RandomInInterval(q0_truncation, 1);
                q1 = RandomInInterval(0, 1 - q0_truncation);
                q2 = RandomInInterval(0, 1 - q0_truncation);
                q3 = RandomInInterval(0, 1 - q0_truncation);
            } while (q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3 < 0.01);
            Vector<double> q = vectorBuilder.DenseOfArray(new double[] { q0, q1, q2, q3 });
            return q / q.Norm(2);
        }

        static Matrix<double>
            RandomRotationMatrix(
                double q0_truncation,
                bool constrain_to_Y_axis,
                int dimension=3
            )
        {
            if (dimension == 2) {
                double t = RandomInInterval(0, 3.1415);
                return matrixBuilder.DenseOfArray(
                    new double[,] {
                        { Math.Cos(t), -Math.Sin(t) },
                        { Math.Sin(t),  Math.Cos(t) }
                    }
                );
            }
            Vector<double> q = RandomQuaternion(q0_truncation);
            if (constrain_to_Y_axis) {
                q = QuaternionConstrainedToY(q);
            }
            return RotationMatrixFromQuaternion(q);
        }

        static Vector<double>
            CrossProduct(Vector<double> u, Vector<double> v)
        {
            return vectorBuilder.DenseOfArray(
                new double[] {
                      u[1] * v[2] - u[2] * v[1],
                    -(u[0] * v[2] - u[2] * v[0]),
                      u[0] * v[1] - u[1] * v[0],
                }
            );
        }

        static Vector<double>
            FindVectorOrthogonalTo(Vector<double> u)
        {
            // assumes a three-dimensional vector u
            Vector<double> e1 = vectorBuilder.DenseOfArray(new double[] { 1, 0, 0 });
            Vector<double> e2 = vectorBuilder.DenseOfArray(new double[] { 0, 1, 0 });
            Vector<double> e3 = vectorBuilder.DenseOfArray(new double[] { 0, 0, 1 });
            Vector<double> candidate1 = CrossProduct(u, e1);
            Vector<double> candidate2 = CrossProduct(u, e2);
            Vector<double> candidate3 = CrossProduct(u, e3);
            double length1 = candidate1.Norm(2);
            double length2 = candidate2.Norm(2);
            double length3 = candidate3.Norm(2);
            return (length1 > Math.Max(length2, length3)) ? candidate1 : ((length2 > length3) ? candidate2 : candidate3);
        }

        static Vector<double>
            QuaternionFromRotationMatrix(Matrix<double> R, bool constrain_rotations_to_Y_axis=false)
        {
            // see https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
            // including comment about the sign of theta
            Vector<double> u = R.Evd().EigenVectors.Column(2);
            if (constrain_rotations_to_Y_axis) {
                if (u[2] < 1E-15) { u[2] = 0; }
                if (u[0] < 1E-15) { u[0] = 0; }
            }
            Vector<double> uperp = FindVectorOrthogonalTo(u);
            Vector<double> uperpperp = CrossProduct(u, uperp);
            int sign = 1;
            if ((R * uperp).DotProduct(uperpperp) < 0)
            {
                sign = -1;
            }
            double cosOfTheta = (R.Trace() - 1) / 2;
            double sinOfHalfTheta = sign * Math.Sqrt((1 - cosOfTheta) / 2);
            double cosOfHalfTheta = Math.Sqrt((1 + cosOfTheta) / 2);
            Vector<double> e = vectorBuilder.DenseOfArray(new double[] { 1, 0, 0, 0 });
            Vector<double> w = vectorBuilder.Dense(4);
            w.SetSubVector(1, 3, u);
            return e * cosOfHalfTheta + w * sinOfHalfTheta;
        }

        static double
            MagnitudeOfRotationFromQuaternionInRadians(Vector<double> q)
        {
            return 2 * Math.Acos(q[0]);
        }

        static double
            MagnitudeOfRotationFromQuaternionInDegrees(Vector<double> q)
        {
            return (180 / 3.141593) * MagnitudeOfRotationFromQuaternionInRadians(q);
        }

        // static void 
        //     IncorporateNewPositionRotationIntoARSessionOrigin(Vector<double> t, Vector<double> q)
        // {
        //     /* notes to self:

        //        let currentP = sessionOrigin.transform.position
        //        let currentR = sessionOrigin.transform.rotation

        //        before: points -> multiply by currentR -> add currentP
        //        after: points -> multiply by currentR -> add currentP -> multiply by R -> add t 

        //        after: points -> R * (currentR * points + currentP) + t

        //        sessionOrigin.transform.position = R * currentP + t
        //        sessionOrigin.transform.rotation = R * currentR */
            
        //     /* actual code: */
        //     var Q = new Quaternion(q[0], q[1], q[2], q[3]);
        //     var transform = sessionOrigin.transform;
        //     transform.position = Q * transform.position + t;
        //     transform.rotation = Q * transform.rotation;
        // }

        static void Main(string[] args)
        {
            Matrix<double> M_points = RandomMatrixInInterval(3, 20, -10, 10);
            Matrix<double> R = RandomRotationMatrix(0.7, true);
            Matrix<double> P_points = R * M_points + RandomMatrixInInterval(M_points.RowCount, M_points.ColumnCount, -0.02, 0.02);

            Vector<double> v = RandomVectorInInterval(M_points.RowCount, 10, 10.01);
            P_points = AddVectorToColumns(P_points, v);

            Tuple<Matrix<double>, Vector<double>, double, int> results = RunJohnICP(
                M_points,
                P_points,
                15,     // max_iterations
                0.01,   // error_per_point_threshold
                0.99,   // lack_of_progress_threshold
                true    // constrain_rotations_to_Y_axis
            );

            System.Console.WriteLine("\nrotation:");
            System.Console.WriteLine(results.Item1);

            System.Console.WriteLine("translation:");
            System.Console.WriteLine(results.Item2);

            System.Console.WriteLine("average error:");
            System.Console.WriteLine(results.Item3);

            System.Console.WriteLine("\nnum iterations:");
            System.Console.WriteLine(results.Item4);

            System.Console.WriteLine("\nR converted to quaternion:");
            var q = QuaternionFromRotationMatrix(results.Item1);
            System.Console.WriteLine(q);

            System.Console.WriteLine("\nmagnitude of rotation in degrees:");
            System.Console.WriteLine(MagnitudeOfRotationFromQuaternionInDegrees(q));
        }
    }
}
