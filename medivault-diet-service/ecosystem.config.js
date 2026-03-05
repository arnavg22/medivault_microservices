module.exports = {
  apps: [
    {
      name: "medivault-diet-service",
      script: "server.py",
      interpreter: "python3",
      env: {
        NODE_ENV: "development",
        PORT: 5001,
      },
      env_production: {
        NODE_ENV: "production",
        PORT: 5001,
      },
    },
  ],
};
