document.addEventListener("DOMContentLoaded", function() {
    const loginForm = document.getElementById('login-form');
    const singin = document.getElementById('login-form');
    const errorMessage = document.getElementById('error-message');
  
    loginForm.addEventListener('submit', function(event) {
      event.preventDefault(); // Prevent form submission
  
      // Get input values
      const username = document.getElementById('username').value;
      const password = document.getElementById('password').value;
  
      // Simple validation
      if (username === 'admin' && password === 'password') {
        // Redirect to dashboard or perform other actions
        alert('Login successful!');
        window.location.href = 'index.html'; // Redirect to dashboard page
      } else {
        errorMessage.textContent = 'Invalid username or password. Please try again.';
      }
    });
  });