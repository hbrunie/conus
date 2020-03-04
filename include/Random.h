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

#ifndef GalSim_Random_H
#define GalSim_Random_H
/**
 * @file Random.h
 *
 * @brief Random-number-generator classes
 *
 * Pseudo-random-number generators with various parent distributions: uniform, Gaussian, binomial,
 * Poisson, Weibull (generalization of Rayleigh and Exponential), and Gamma, all living within the
 * galsim namespace.
 *
 * Wraps Boost.Random classes in a way that lets us swap Boost RNG's without affecting client code.
 */

#include <sstream>

namespace galsim {

    /**
     * @brief Base class for all the various Deviates.
     *
     * This holds the essential random number generator that all the other classes use.
     *
     * All deviates have three constructors that define different ways of setting up
     * the random number generator.
     *
     * 1) Only the arguments particular to the derived class (e.g. mean and sigma for
     *    GaussianDeviate).  In this case, a new random number generator is created and
     *    it is seeded using the computer's microsecond counter.
     *
     * 2) Using a particular seed as the first argument to the constructor.
     *    This will also create a new random number generator, but seed it with the
     *    provided value.
     *
     * 3) Passing another BaseDeviate as the first arguemnt to the constructor.
     *    This will make the new Deviate share the same underlying random number generator
     *    with the other Deviate.  So you can make one Deviate (of any type), and seed
     *    it with a particular deterministic value.  Then if you pass that Deviate
     *    to any other one you make, they will all be using the same rng and have a
     *    particular deterministic series of values.  (It doesn't have to be the first
     *    one -- any one you've made later can also be used to seed a new one.)
     *
     * There is not much you can do with something that is only known to be a BaseDeviate
     * rather than one of the derived classes other than construct it and change the
     * seed, and use it as an argument to pass to other Deviate constructors.
     */
    class BaseDeviate
    {
    public:
        /**
         * @brief Construct and seed a new BaseDeviate, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed A long-integer seed for the RNG.
         */
        explicit BaseDeviate(long lseed);

        /**
         * @brief Construct a new BaseDeviate, sharing the random number generator with rhs.
         */
        BaseDeviate(const BaseDeviate& rhs);

        /**
         * @brief Construct a new BaseDeviate from a serialization string
         */
        BaseDeviate(const char* str_c);

        /**
         * @brief Destructor
         *
         * Only deletes the underlying RNG if this is the last one using it.
         */
        virtual ~BaseDeviate() {}

        /// @brief return a serialization string for this BaseDeviate
        std::string serialize();

        /**
         * @brief Construct a duplicate of this BaseDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        BaseDeviate duplicate()
        { return BaseDeviate(serialize().c_str()); }

        /**
         * @brief Return a string that can act as the repr in python
         */
        std::string repr() { return make_repr(true); }

        /**
         * @brief Return a string that can act as the str in python
         *
         * For this we use the same thing as the repr, but omit the (verbose!) seed parameter.
         */
        std::string str() { return make_repr(false); }

        /**
         * @brief Re-seed the PRNG using specified seed
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed A long-integer seed for the RNG.
         *
         * Note that this will reseed all Deviates currently sharing the RNG with this one.
         */
        virtual void seed(long lseed);

        /**
         * @brief Like seed(lseed), but severs the relationship between other Deviates.
         *
         * Other Deviates that had been using the same RNG will be unaffected, while this
         * Deviate will obtain a fresh RNG seed according to lseed.
         */
        void reset(long lseed);

        /**
         * @brief Make this object share its random number generator with another Deviate.
         *
         * It discards whatever rng it had been using and starts sharing the one held by dev.
         */
        void reset(const BaseDeviate& dev);

        /**
         * @brief Clear the internal cache of the rng object.
         *
         * Sometimes this is required to get two sequences synced up if the other one
         * is reseeded.  e.g. GaussianDeviate generates two deviates at a time for efficiency,
         * so if you don't do this, and there is still an internal cached value, you'll get
         * that rather than a new one generated with the new seed.
         *
         * As far as I know, GaussianDeviate is the only one to require this, but just in
         * case something changes about how boost implements any of these deviates, I overload
         * the virtual function for all of them and call the distribution's reset() method.
         */
        virtual void clearCache() {}

        /**
         * @brief Discard some number of values from the random number generator.
         */
        void discard(int n);

        /**
         * @brief Get a random value in its raw form as a long integer.
         */
        long raw();

        /**
         * @brief Draw a new random number from the distribution
         *
         * This is invalid for a BaseDeviate object that is not a derived class.
         * However, we don't make it pure virtual, since we want to be able to make
         * BaseDeviate objects as a direct way to define a common seed for other Deviates.
         */
        double operator()()
        { return generate1(); }

        // This is the virtual function that is overridden in subclasses.
        virtual double generate1()
        { throw std::runtime_error("Cannot draw random values from a pure BaseDeviate object."); }

        /**
         * @brief Draw N new random numbers from the distribution and save the values in
         * an array
         *
         * @param N     The number of values to draw
         * @param data  The array into which to write the values
         */
        void generate(int N, double* data);

        /**
         * @brief Draw N new random numbers from the distribution and add them to the values in
         * an array
         *
         * @param N     The number of values to draw
         * @param data  The array into which to add the values
         */
        void addGenerate(int N, double* data);

   protected:
        struct BaseDeviateImpl;
        shared_ptr<BaseDeviateImpl> _impl;

        /// Helper to make the repr with or without the (lengthy!) seed item.
        virtual std::string make_repr(bool incl_seed);

        /**
         * @brief Private routine to seed with microsecond counter from time-of-day structure.
         */
        void seedtime();

        /**
         * @brief Private routine to seed using /dev/random.  This will throw an exception
         * if this is not possible.
         */
        void seedurandom();
    };

    /**
     * @brief Pseudo-random number generator with uniform distribution in interval [0.,1.).
     */
    class UniformDeviate : public BaseDeviate
    {
    public:
        /** @brief Construct and seed a new UniformDeviate, using the provided value as seed.
         *
         * If lseed == 0, this means to use a random seed from the system: either /dev/urandom
         * if possible, or the time of day otherwise.  Note that in the latter case, the
         * microsecond counter is the seed, so BaseDeviates constructed in rapid succession may
         * not be independent.
         *
         * @param[in] lseed A long-integer seed for the RNG.
         */
        UniformDeviate(long lseed);

        /// @brief Construct a new UniformDeviate, sharing the random number generator with rhs.
        UniformDeviate(const BaseDeviate& rhs);

        /// @brief Construct a copy that shares the RNG with rhs.
        UniformDeviate(const UniformDeviate& rhs);

        /// @brief Construct a new UniformDeviate from a serialization string
        UniformDeviate(const char* str_c);

        /**
         * @brief Construct a duplicate of this UniformDeviate object.
         *
         * Both this and the returned duplicate will produce identical sequences of values.
         */
        UniformDeviate duplicate()
        { return UniformDeviate(serialize().c_str()); }

        /**
         * @brief Draw a new random number from the distribution
         *
         * @return A uniform deviate in the interval [0.,1.)
         */
        double generate1();

        /**
         * @brief Clear the internal cache
         */
        void clearCache();

    protected:
        std::string make_repr(bool incl_seed);

    private:
        struct UniformDeviateImpl;
        shared_ptr<UniformDeviateImpl> _devimpl;
    };

}  // namespace galsim

#endif
