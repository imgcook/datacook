#include <random>
#include <stdlib.h>
#include <emscripten/bind.h>

using namespace emscripten;
// refer to https://gist.github.com/sftrabbit/5068941

class beta_distribution
{
  public:
    class param_type
    {
      public:
        typedef beta_distribution distribution_type;

        explicit param_type(double a = 2.0, double b = 2.0)
          : a_param(a), b_param(b) { }

        double a() const { return a_param; }
        double b() const { return b_param; }
  
      private:
        double a_param, b_param;
    };

    explicit beta_distribution(double a = 2.0, double b = 2.0)
      : a_gamma(a), b_gamma(b) { 
        rng.seed(rand());
      }

    void param(double a, double b)
    {
      a_gamma = gamma_dist_type(a);
      b_gamma = gamma_dist_type(b);
    }

    void seed(uint32_t seed) {
      rng.seed(seed);
    }

    double generate() {
      return generate_internal(a_gamma, b_gamma);
    }

    double min() const { return 0.0; }
    double max() const { return 1.0; }

    double a() const { return a_gamma.alpha(); }
    double b() const { return b_gamma.alpha(); }

  private:
    typedef std::gamma_distribution<double> gamma_dist_type;

    std::mt19937 rng;

    gamma_dist_type a_gamma, b_gamma;

    double generate_internal(
      gamma_dist_type& x_gamma,
      gamma_dist_type& y_gamma)
    { 
      double x = x_gamma(rng);
      return x / (x + y_gamma(rng));
    }
};

// Binding code
EMSCRIPTEN_BINDINGS(my_class_example) {
  class_<beta_distribution>("Beta")
    .constructor<double, double>()
    .constructor<>()
    .function("generate", &beta_distribution::generate)
    .function("setParam", &beta_distribution::param)
    .function("setSeed", &beta_distribution::seed)
    .property("a", &beta_distribution::a)
    .property("b", &beta_distribution::a)
    ;
}
