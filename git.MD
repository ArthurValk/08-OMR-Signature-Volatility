# Working with Git

Git is a distributed version control system that tracks changes in your code over time.

## Getting Started

1. **Clone the repository** to your local machine:
   ```
   git clone <repository-url>
   ```

## Adding Changes

1. **Create a new branch** for your changes:
   ```
   git checkout -b feature/your-feature-name
   ```
   Depending on your IDE, you might be able to create and switch branches directly from the interface, which is often easier.

2. **Make your changes** to the code

3. **Stage and commit** your changes:
   ```
   git add .
   git commit -m "Description of your changes"
   ```
   Most modern IDEs have built-in Git support that allows you to stage and commit changes through a graphical interface. Google "how to use git in your IDE of choice for specific instructions.

4. **Push your branch** to GitHub:
   ```
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on the GitHub website when ready to merge your changes into the main branch

## Important:

**Do not commit directly to the main branch!**

- Always create a new branch for your changes
- Don't push directly to main - use Pull Requests instead
- This keeps the main branch stable