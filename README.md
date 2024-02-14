# heimdal

## Deployment instructions

- Populate your `.env` file by renaming the `.env.example` file to `.env` and filling in the values
- Deploy those secrets into the namespace by running `kubectl create secret generic heimdal-creds --from-env-file .env`
- Deploy the application by running `kubectl apply -k .`

