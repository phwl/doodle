# 1. Use the official, daily-maintained stable light image for JDK 17 & SBT 1.x
FROM sbtscala/scala-sbt:eclipse-temurin-17_1.x

# 2. Install Verilator and its C++ build compilation tools
USER root
RUN apt-get update && apt-get install -y \
    verilator \
    make \
    g++ \
    flex \
    bison \
    libfl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy build settings first to cache dependencies safely
COPY project/build.properties ./project/
COPY build.sbt .

# 4. Warm up SBT (Resolves your Chisel 6.6.0 packages ahead of time)
RUN sbt update

# 5. Copy the remaining application and test files
COPY . .

# 6. Pre-compile testbenches to verify setup
RUN sbt test:compile

# 7. Default container execution rule
CMD ["sbt", "test"]

