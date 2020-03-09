/* -*- c++ -*-
 * Copyright (c) 2012-2019 by the GalSim developers team on GitHub
 * https://github.com/GalSim-developers
 *
 * This file is part of GalSim: The modular galaxy image simulation toolkit.
 * https://github.com/GalSim-developers/GalSim
 *
 * GalSim is free software: redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions, and the disclaimer given in the accompanying LICENSE
 *    file.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions, and the disclaimer given in the documentation
 *    and/or other materials provided with the distribution.
 */

#include <sys/time.h>
#include <fcntl.h>
#include <string>
#include <vector>
#include <sstream>
#include <unistd.h>
#include "Random.h"

// #include "galsim/IgnoreWarnings.h"

namespace galsim {

    struct BaseDeviate::BaseDeviateImpl
    {
        // Note that this class could be templated with the type of Boost.Random generator that
        // you want to use instead of mt19937
        typedef std::mt19937 rng_type;
        BaseDeviateImpl() : _rng(new rng_type) {}
        std::shared_ptr<rng_type> _rng;
    };

    BaseDeviate::BaseDeviate(long lseed) :
        _impl(new BaseDeviateImpl())
    { seed(lseed); }

    BaseDeviate::BaseDeviate(const BaseDeviate& rhs) :
        _impl(rhs._impl)
    {}

    BaseDeviate::BaseDeviate(const char* str_c) :
        _impl(new BaseDeviateImpl())
    {
        if (str_c == NULL) {
            seed(0);
        } else {
            std::string str(str_c);
            std::istringstream iss(str);
            iss >> *_impl->_rng;
        }
    }

    std::string BaseDeviate::serialize()
    {
        // When serializing, we need to make sure there is no cache being stored
        // by the derived class.
        clearCache();
        std::ostringstream oss;
        oss << *_impl->_rng;
        return oss.str();
    }

    void BaseDeviate::seedurandom()
    {
        // This implementation shamelessly taken from:
        // http://stackoverflow.com/questions/2572366/how-to-use-dev-random-or-urandom-in-c
        int randomData = open("/dev/urandom", O_RDONLY);
        int myRandomInteger;
        size_t randomDataLen = 0;
        while (randomDataLen < sizeof myRandomInteger)
        {
            ssize_t result = read(randomData, ((char*)&myRandomInteger) + randomDataLen,
                                  (sizeof myRandomInteger) - randomDataLen);
            if (result < 0)
                throw std::runtime_error("Unable to read from /dev/urandom");
            randomDataLen += result;
        }
        close(randomData);
        _impl->_rng->seed(myRandomInteger);
    }

    void BaseDeviate::seedtime()
    {
        struct timeval tp;
        gettimeofday(&tp,NULL);
        _impl->_rng->seed(tp.tv_usec);
    }

    void BaseDeviate::seed(long lseed)
    {
        if (lseed == 0) {
            try {
                seedurandom();
            } catch(...) {
                // If urandom is not possible, revert to using the time
                seedtime();
            }
        } else {
            // We often use sequential seeds for our RNG's (so we can be sure that runs on multiple
            // processors are deterministic).  The Boost Mersenne Twister is supposed to work with
            // this kind of seeding, having been updated in April 2005 to address an issue with
            // precisely this sort of re-seeding.
            // (See http://www.boost.org/doc/libs/1_51_0/boost/random/mersenne_twister.hpp).
            // The issue itself is described briefly here:
            // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html,
            // and in more detail for an algorithm tt880 that is apparently a 'little cousin' to
            // the Mersenne Twister: http://random.mat.sbg.ac.at/news/seedingTT800.html
            //
            // The worry is that updates to the methods claim improvements to the behaviour of
            // close (in a bitwise sense) patterns, but we have not found ready quantified data.
            //
            // So just to be sure, we send the initial seed through a _different_ random number
            // generator for 2 iterations before using it to seed the RNG we will actually use.
            // This may not be necessary, but it's not much of a performance hit (only occurring on
            // the initial seed of each rng), it can't hurt, and it makes Barney and Mike somewhat
            // less disquieted.  :)

            std::ranlux48 alt_rng(lseed);
            alt_rng.discard(2);
            _impl->_rng->seed(alt_rng());
        }
        clearCache();
    }

    void BaseDeviate::reset(long lseed)
    { _impl.reset(new BaseDeviateImpl()); seed(lseed); }

    void BaseDeviate::reset(const BaseDeviate& dev)
    { _impl = dev._impl; clearCache(); }

    void BaseDeviate::discard(int n)
    { _impl->_rng->discard(n); }

    long BaseDeviate::raw()
    { return (*_impl->_rng)(); }

    void BaseDeviate::generate(int N, double* data)
    {
        for (int i=0; i<N; ++i) data[i] = (*this)();
    }

    void BaseDeviate::addGenerate(int N, double* data)
    {
        for (int i=0; i<N; ++i) data[i] += (*this)();
    }

    // Next two functions shamelessly stolen from
    // http://stackoverflow.com/questions/236129/split-a-string-in-c
    std::vector<std::string>& split(const std::string& s, char delim,
                                    std::vector<std::string>& elems)
    {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }
        return elems;
    }

    std::vector<std::string> split(const std::string& s, char delim)
    {
        std::vector<std::string> elems;
        split(s, delim, elems);
        return elems;
    }

    std::string seedstring(const std::vector<std::string>& seed)
    {
        std::ostringstream oss;
        int nseed = seed.size();
        oss << "seed='";
        for (int i=0; i < 3; i++) oss << seed[i] << ' ';
        oss << "...";
        for (int i=nseed-3; i < nseed; i++) oss << ' ' << seed[i];
        oss << "'";
        return oss.str();
    }

    std::string BaseDeviate::make_repr(bool incl_seed)
    {
        // Remember: Don't start with nothing!  See discussion in FormatAndThrow in Std.h
        std::ostringstream oss(" ");
        oss << "galsim.BaseDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' '));
        oss<<")";

        return oss.str();
    }

    struct UniformDeviate::UniformDeviateImpl
    {
        UniformDeviateImpl() : _urd(0., 1.) {}
        std::uniform_real_distribution<> _urd;
    };

    UniformDeviate::UniformDeviate(long lseed) :
        BaseDeviate(lseed), _devimpl(new UniformDeviateImpl()) {}

    UniformDeviate::UniformDeviate(const BaseDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(new UniformDeviateImpl()) {}

    UniformDeviate::UniformDeviate(const UniformDeviate& rhs) :
        BaseDeviate(rhs), _devimpl(rhs._devimpl) {}

    UniformDeviate::UniformDeviate(const char* str_c) :
        BaseDeviate(str_c), _devimpl(new UniformDeviateImpl()) {}

    void UniformDeviate::clearCache() { _devimpl->_urd.reset(); }

    double UniformDeviate::generate1()
    { return _devimpl->_urd(*this->_impl->_rng); }

    std::string UniformDeviate::make_repr(bool incl_seed)
    {
        std::ostringstream oss(" ");
        oss << "galsim.UniformDeviate(";
        if (incl_seed) oss << seedstring(split(serialize(), ' '));
        oss<<")";
        return oss.str();
    }

}
