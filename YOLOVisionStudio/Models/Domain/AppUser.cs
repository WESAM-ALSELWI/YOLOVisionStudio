using System;

namespace YOLOVisionStudio.Models.Domain
{
    /// <summary>
    /// Represents the user roles in the system.
    /// </summary>
    public enum UserRole
    {
        /// <summary>System Administrator</summary>
        Admin,
        /// <summary>AI Researcher</summary>
        Researcher,
        /// <summary>Standard User</summary>
        User
    }

    /// <summary>
    /// Represents a local studio user. Login is not required in Studio Mode.
    /// </summary>
    public class AppUser
    {
        /// <summary>Gets or sets the user ID.</summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>Gets or sets the username.</summary>
        public string UserName { get; set; } = string.Empty;

        /// <summary>Gets or sets the email address.</summary>
        public string? Email { get; set; }

        /// <summary>Gets or sets the display name.</summary>
        public string? DisplayName { get; set; }

        /// <summary>Gets or sets the user role.</summary>
        public UserRole Role { get; set; } = UserRole.User;

        /// <summary>Gets or sets the creation timestamp.</summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>Gets or sets the last login timestamp.</summary>
        public DateTime? LastLoginAt { get; set; }

        /// <summary>Gets or sets the avatar URL.</summary>
        public string? AvatarUrl { get; set; }

        /// <summary>Gets or sets the user's bio.</summary>
        public string? Bio { get; set; }
    }
}