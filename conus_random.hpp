#include "Random.h"


class ConusUniform : public galsim::BaseDeviate {

    public:

        ConusUniform(long lseed, int N);

        double generate1();

    private:
        int buf_len;
        int buf_ptr;

        std::unique_ptr<double> buf;

        void fill_buff();
};
