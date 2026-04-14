const API_URL = 'http://127.0.0.1:8000';

// DOM Elements
const studentTableBody = document.getElementById('studentListBody');
const studentForm = document.getElementById('studentForm');
const studentModal = document.getElementById('studentModal');
const addStudentBtn = document.getElementById('addStudentBtn');
const closeModalBtn = document.getElementById('closeModal');
const cancelModalBtn = document.getElementById('cancelModal');
const modalTitle = document.getElementById('modalTitle');
const totalStudentsCount = document.getElementById('totalStudents');
const noDataContainer = document.getElementById('noData');
const searchInput = document.getElementById('searchInput');

let students = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    fetchStudents();
});

// Fetch All Students
async function fetchStudents() {
    try {
        const response = await fetch(`${API_URL}/students-all/`);
        students = await response.json();
        renderStudents(students);
        updateStats();
    } catch (error) {
        showToast('Error fetching students', 'error');
    }
}

// Render Students Table
function renderStudents(data) {
    studentTableBody.innerHTML = '';
    
    if (data.length === 0) {
        noDataContainer.classList.remove('hidden');
    } else {
        noDataContainer.classList.add('hidden');
        data.forEach(student => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>
                    <div class="student-name">
                        <strong>${student.name}</strong>
                    </div>
                </td>
                <td>${student.email}</td>
                <td><span class="badge">${student.age}</span></td>
                <td>${student.course}</td>
                <td class="actions">
                    <button class="btn-icon btn-edit" onclick="editStudent(${student.id})">
                        <i data-lucide="edit-3"></i>
                    </button>
                    <button class="btn-icon btn-delete" onclick="deleteStudent(${student.id})">
                        <i data-lucide="trash-2"></i>
                    </button>
                </td>
            `;
            studentTableBody.appendChild(row);
        });
        lucide.createIcons();
    }
}

// Update Dashboard Stats
function updateStats() {
    totalStudentsCount.textContent = students.length;
}

// Search Functionality
searchInput.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const filtered = students.filter(s => 
        s.name.toLowerCase().includes(term) || 
        s.email.toLowerCase().includes(term) || 
        s.course.toLowerCase().includes(term)
    );
    renderStudents(filtered);
});

// Modal Logic
addStudentBtn.onclick = () => {
    modalTitle.textContent = 'Add New Student';
    studentForm.reset();
    document.getElementById('studentId').value = '';
    studentModal.classList.add('active');
};

const hideModal = () => studentModal.classList.remove('active');
closeModalBtn.onclick = hideModal;
cancelModalBtn.onclick = hideModal;

window.onclick = (event) => {
    if (event.target == studentModal) hideModal();
};

// Form Submission (Add/Update)
studentForm.onsubmit = async (e) => {
    e.preventDefault();
    
    const id = document.getElementById('studentId').value;
    const studentData = {
        name: document.getElementById('name').value,
        email: document.getElementById('email').value,
        age: parseInt(document.getElementById('age').value),
        course: document.getElementById('course').value
    };

    try {
        let response;
        if (id) {
            // Update
            response = await fetch(`${API_URL}/update-students/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(studentData)
            });
        } else {
            // Create
            response = await fetch(`${API_URL}/create-student/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(studentData)
            });
        }

        if (response.ok) {
            showToast(id ? 'Student updated successfully' : 'Student added successfully', 'success');
            hideModal();
            fetchStudents();
        } else {
            const error = await response.json();
            showToast(error.detail || 'Operation failed', 'error');
        }
    } catch (error) {
        showToast('Network error', 'error');
    }
};

// Edit Student
window.editStudent = async (id) => {
    const student = students.find(s => s.id === id);
    if (!student) return;

    modalTitle.textContent = 'Edit Student';
    document.getElementById('studentId').value = student.id;
    document.getElementById('name').value = student.name;
    document.getElementById('email').value = student.email;
    document.getElementById('age').value = student.age;
    document.getElementById('course').value = student.course;
    
    studentModal.classList.add('active');
};

// Delete Student
window.deleteStudent = async (id) => {
    if (!confirm('Are you sure you want to delete this student?')) return;

    try {
        const response = await fetch(`${API_URL}/delete-students/${id}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            showToast('Student deleted successfully', 'success');
            fetchStudents();
        } else {
            showToast('Failed to delete student', 'error');
        }
    } catch (error) {
        showToast('Network error', 'error');
    }
};

// Toast Notifications
function showToast(message, type) {
    const toast = document.getElementById('toast');
    const icon = document.getElementById('toastIcon');
    const msg = document.getElementById('toastMessage');

    toast.className = `toast active ${type}`;
    msg.textContent = message;
    
    const iconName = type === 'success' ? 'check-circle' : 'alert-circle';
    icon.setAttribute('data-lucide', iconName);
    lucide.createIcons();

    setTimeout(() => {
        toast.classList.remove('active');
    }, 3000);
}
