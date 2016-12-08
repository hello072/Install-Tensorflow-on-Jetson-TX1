Install-Tensorflow-and-Caffe-on-Jetson-TX1
========
##JetPack2.3을 설치한 TX1에 Tensorflow 설치하기!
--------------
JetPack의 도움으로 ubuntu16.04,cuda8, cudnn5.1 등이 설치되었다고 가정합니다.

편의를 위해 주 작업장 /home/ubuntu/Downloads 를 HOMEPATH로 하겠습니다.

일단 기본 패키지들을 설치합니다.(크롬같은건 당장은 필요없지만 그냥 ㅎㅎ)
<pre><code>sudo apt-get install chromium-browser
sudo apt-get install python-pip
sudo apt-get install libopencv4tegra-python
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer
sudo apt-get install git zip unzip autoconf automake libtool curl zlib1g-dev maven swig
</code></pre>
HOMEPATH로 가서 다음 명령어들을 통해 protobuf를 설치합니다.
<pre><code>git clone https://github.com/google/protobuf.git
cd protobuf
git checkout master
./autogen.sh
git checkout d5fb408d
./configure --prefix=/usr
make -j 4
sudo make install
cd java
mvn package
</code></pre>
다시 HOMEPATH로 가서 bazel을 내려받고 작업합니다.
<pre><code>git clone https://github.com/bazelbuild/bazel.git
cd bazel
git checkout 0.2.1
cp /usr/bin/protoc third_party/protobuf/protoc-linux-arm32.exe
cp ../protobuf/java/target/protobuf-java-3.0.0-beta-2.jar third_party/protobuf/protobuf-java-3.0.0-beta-1.jar
</code></pre>
HOMEPATH/bazel/src/main/java/com/google/devtools/build/lib/util/CPU.java 파일을 열고 다음과 같이 수정합니다. (- 가 붙은 줄을 지우고 +가 붙은 줄을 붙입니다.)
<pre><code>@@ -25,7 +25,7 @@ import java.util.Set;
 public enum CPU {
   X86_32("x86_32", ImmutableSet.of("i386", "i486", "i586", "i686", "i786", "x86")),
   X86_64("x86_64", ImmutableSet.of("amd64", "x86_64", "x64")),
-  ARM("arm", ImmutableSet.of("arm", "armv7l")),
+  ARM("arm", ImmutableSet.of("arm", "armv7l", "aarch64")),
   UNKNOWN("unknown", ImmutableSet.<String>of());
</code></pre>
HOMEPATH/bazel로 가서 다음 명령어를 통해 bazel을 설치합니다.
<pre><code>./compile.sh
sudo cp output/bazel /usr/local/bin
</code></pre>
HOMEPATH로 가서 다음 명령어를 통해 tensorflow를 설치합니다.
<pre><code>git clone -b r0.9 https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
bazel build -c opt --local_resources 2048,.5,1.0 --config=cuda //tensorflow/tools/pip_package:build_pip_package
</code></pre>
아마 잘 되는 것 같다가 실패할텐데, 일부러 실패한 것입니다. 이제 다음 명령어를 입력하세요! (여기서 폴더명 f596b50637e57f31ad9bfc386482aa22은 사람마다 다릅니다. 각자 폴더 탐험을 통해 폴더명을 알아내세요!)
<pre><code>cd ~
wget -O config.guess 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD'
wget -O config.sub 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD'
cp config.guess ./.cache/bazel/_bazel_ubuntu/f596b50637e57f31ad9bfc386482aa22/external/farmhash_archive/farmhash-34c13ddfab0e35422f4c3979f360635a8c050260/config.guess
cp config.sub ./.cache/bazel/_bazel_ubuntu/f596b50637e57f31ad9bfc386482aa22/external/farmhash_archive/farmhash-34c13ddfab0e35422f4c3979f360635a8c050260/config.sub
</code></pre>
HOMEPATH/tensorflow/tensorflow/core/kernels/BUILD 파일을 열어서 다음과 같이 수정하세요.
<pre><code>@@ -985,7 +985,7 @@ tf_kernel_libraries(
         "reduction_ops",
         "segment_reduction_ops",
         "sequence_ops",
-        "sparse_matmul_op",
+        #DC "sparse_matmul_op",
     ],
     deps = [
         ":bounds_check",
</code></pre>
In HOMEPATH/tensorflow/tensorflow/python/BUILD 파일을 열어서 다음과 같이 수정하세요.
<pre><code>@@ -1110,7 +1110,7 @@ medium_kernel_test_list = glob([
     "kernel_tests/seq2seq_test.py",
     "kernel_tests/slice_op_test.py",
     "kernel_tests/sparse_ops_test.py",
-    "kernel_tests/sparse_matmul_op_test.py",
+    #DC "kernel_tests/sparse_matmul_op_test.py",
     "kernel_tests/sparse_tensor_dense_matmul_op_test.py",
 ])
</code></pre>
In HOMEPATH/tensorflow/tensorflow/core/kernels/cwise_op_gpu_select.cu.cc 파일을 열어서 다음과 같이 수정하세요.
<pre><code>@@ -43,8 +43,14 @@ struct BatchSelectFunctor<GPUDevice, T> {
     const int all_but_batch = then_flat_outer_dims.dimension(1);

 #if !defined(EIGEN_HAS_INDEX_LIST)
-    Eigen::array<int, 2> broadcast_dims{{ 1, all_but_batch }};
-    Eigen::Tensor<int, 2>::Dimensions reshape_dims{{ batch, 1 }};
+    //DC Eigen::array<int, 2> broadcast_dims{{ 1, all_but_batch }};
+    Eigen::array<int, 2> broadcast_dims;
+    broadcast_dims[0] = 1;
+    broadcast_dims[1] = all_but_batch;
+    //DC Eigen::Tensor<int, 2>::Dimensions reshape_dims{{ batch, 1 }};
+    Eigen::Tensor<int, 2>::Dimensions reshape_dims;
+    reshape_dims[0] = batch;
+    reshape_dims[1] = 1;
 #else
     Eigen::IndexList<Eigen::type2index<1>, int> broadcast_dims;
     broadcast_dims.set(1, all_but_batch);
</code></pre>
In HOMEPATH/tensorflow/tensorflow/core/kernels/sparse_tensor_dense_matmul_op_gpu.cu.cc 파일을 열어서 다음과 같이 수정하세요.
<pre><code>@@ -104,9 +104,17 @@ struct SparseTensorDenseMatMulFunctor<GPUDevice, T, ADJ_A, ADJ_B> {
     int n = (ADJ_B) ? b.dimension(0) : b.dimension(1);

 #if !defined(EIGEN_HAS_INDEX_LIST)
-    Eigen::Tensor<int, 2>::Dimensions matrix_1_by_nnz{{ 1, nnz }};
-    Eigen::array<int, 2> n_by_1{{ n, 1 }};
-    Eigen::array<int, 1> reduce_on_rows{{ 0 }};
+    //DC Eigen::Tensor<int, 2>::Dimensions matrix_1_by_nnz{{ 1, nnz }};
+    Eigen::Tensor<int, 2>::Dimensions matrix_1_by_nnz;
+    matrix_1_by_nnz[0] = 1;
+    matrix_1_by_nnz[1] = nnz;
+    //DC Eigen::array<int, 2> n_by_1{{ n, 1 }};
+    Eigen::array<int, 2> n_by_1;
+    n_by_1[0] = n;
+    n_by_1[1] = 1;
+    //DC Eigen::array<int, 1> reduce_on_rows{{ 0 }};
+    Eigen::array<int, 1> reduce_on_rows;
+    reduce_on_rows[0] = 0;
 #else
     Eigen::IndexList<Eigen::type2index<1>, int> matrix_1_by_nnz;
     matrix_1_by_nnz.set(1, nnz);
</code></pre>
In HOMEPATH/tensorflow/tensorflow/stream_executor/cuda/cuda_blas.cc 파일을 열어서 다음과 같이 수정하세요.
<pre><code>@@ -25,6 +25,12 @@ limitations under the License.
 #define EIGEN_HAS_CUDA_FP16
 #endif
+#if CUDA_VERSION >= 8000
+#define SE_CUDA_DATA_HALF CUDA_R_16F
+#else
+#define SE_CUDA_DATA_HALF CUBLAS_DATA_HALF
+#endif
 #include "tensorflow/stream_executor/cuda/cuda_blas.h"

@@ -1680,10 +1686,10 @@ bool CUDABlas::DoBlasGemm(
   return DoBlasInternal(
       dynload::cublasSgemmEx, stream, true /* = pointer_mode_host */,
       CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
-      CUDAMemory(a), CUBLAS_DATA_HALF, lda,
-      CUDAMemory(b), CUBLAS_DATA_HALF, ldb,
+      CUDAMemory(a), SE_CUDA_DATA_HALF, lda,
+      CUDAMemory(b), SE_CUDA_DATA_HALF, ldb,
       &beta,
-      CUDAMemoryMutable(c), CUBLAS_DATA_HALF, ldc);
+      CUDAMemoryMutable(c), SE_CUDA_DATA_HALF, ldc);
 #else
   LOG(ERROR) << "fp16 sgemm is not implemented in this cuBLAS version "
              << "(need at least CUDA 7.5)";
</code></pre>
In HOMEPATH/tensorflow/tensorflow/stream_executor/cuda/cuda_gpu_executor.cc 파일을 열어서 다음과 같이 수정하세요.
<pre><code>@@ -888,6 +888,9 @@ CudaContext* CUDAExecutor::cuda_context() { return context_; }
 // For anything more complicated/prod-focused than this, you'll likely want to
 // turn to gsys' topology modeling.
 static int TryToReadNumaNode(const string &pci_bus_id, int device_ordinal) {
+  // DC - make this clever later. ARM has no NUMA node, just return 0
+  LOG(INFO) << "ARM has no NUMA node, hardcoding to return zero";
+  return 0;
 #if defined(__APPLE__)
   LOG(INFO) << "OS X does not support NUMA - returning NUMA node zero";
   return 0;
</code></pre>
아마 지금쯤 메모리 사용량이 좀 클텐데, refresh 차원에서 재부팅을 해봅시다! (효과가 있는지는 모르겠으나 ㅎㅎ 재량껏...)

In HOMEPATH/tensorflow 에서 다음 명령어들을 통해 tensorflow를 진짜로 설치합니다.

메모리가 적은 tx1 특성상 --local_resources 2048,.5,1.0 명령어를 통해 메모리사용 제한을 걸어주는 것이 중요합니다.

설치 도중 튕기는 경우가 간혹 있는데, 다시 시도하면 설치가 계속 진행됩니다. 될 때까지 고고!
<pre><code>./configure
bazel build -c opt --local_resources 2048,.5,1.0 --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-0.9.0-py2-none-any.whl
</code></pre>
이제 터미널에서 python을 켜서 import tensorflow로 확인을 해보세요~ CUDA 관련 메시지가 뜨면서 import되면 정상!

##JetPack2.3을 설치한 TX1에 Caffe 설치하기!
--------------
JetPack의 도움으로 ubuntu16.04,cuda8, cudnn5.1 등이 설치되었다고 가정합니다.
OpenCV 혹은 OpenCV4Tegra가 설치되어 있어야 합니다.
편의를 위해 주 작업장 /home/ubuntu/Downloads 를 HOMEPATH로 하겠습니다.
HOMEPATH에서 다음 명령을 통해 Caffe 설치 스크립트를 다운받습니다.
<pre><code>git clone https://github.com/jetsonhacks/installCaffeJTX1.git
cd installCaffeJTX1
</code></pre>
installCaffe.sh 파일을 열고 make -j4 를 make -j3 으로 바꿉니다.
그 다음 명령어를 통해 Caffe를 설치합니다.
<pre><code>./installCaffe.sh
</code></pre>
/home 폴더에 caffe가 생겼다면 정상!
